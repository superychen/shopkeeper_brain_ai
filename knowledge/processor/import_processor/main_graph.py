"""导入模块的总流程入口，建议先读本文件，再沿节点进入 service 和 repository。

执行走向（箭头表示前一个节点成功返回后才执行下一个节点）：
    CLI/API → run_import_graph → EntryNode
        PDF：PdfToMdNode → MdToImgNode
        Markdown：MdToImgNode
    → DocumentSplitNode → ItemNameRecognitionNode
    → BgeEmbeddingChunksNode → MilvusImportNode → END → 返回最终状态。

数据走向：文件路径 → Markdown → 图片处理后的正文 → chunks
→ 商品名回填 → 每条切片的 dense/sparse 向量 → Milvus 主键及写入数量。
每个节点只返回需要更新的状态字段，LangGraph 将它们合并进 ImportGraphState。
节点抛异常会中断串行调用；异常不会被当作成功状态继续传给下一个节点。
"""

import logging
import threading
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.service.import_validation import validate_config
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.nodes.document_split_node import (
    DocumentSplitNode,
)
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import (
    ItemNameRecognitionNode,
)
from knowledge.processor.import_processor.nodes.md_to_img_node import MdToImgNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state

logger = logging.getLogger("import.main_graph")
_import_lock = threading.Lock()

ImportRoute = Literal["pdf", "markdown"]


def _route_import_file(state: ImportGraphState) -> ImportRoute:
    """根据入口节点写入的互斥标志选择下一节点。"""
    is_pdf_enabled = state.get("is_pdf_read_enabled") is True
    is_markdown_enabled = state.get("is_md_read_enabled") is True

    # 两个分支必须且只能启用一个，避免错误状态被静默路由到任意节点。
    if is_pdf_enabled == is_markdown_enabled:
        raise StateFieldError(
            node_name=EntryNode.name,
            field_name="is_pdf_read_enabled/is_md_read_enabled",
            expected_type=bool,
            message="PDF 与 Markdown 读取标志必须且只能启用一个",
        )

    return "pdf" if is_pdf_enabled else "markdown"


def build_import_graph(
    config: ImportConfig | None = None,
    observer=None,
) -> CompiledStateGraph:
    """组装并编译可执行图；此处定义连接关系，不实际执行导入。

    config 注入所有节点，使切分、编码和写库使用同一套配置。
    返回的 CompiledStateGraph 由 run_import_graph 调用 invoke 执行。
    """
    # 第一步：创建节点对象。BaseNode.__call__ 统一记录日志，子类 process 负责业务校验。
    entry_node = EntryNode(config=config)
    pdf_to_md_node = PdfToMdNode(config=config)
    md_to_img_node = MdToImgNode(config=config)
    document_split_node = DocumentSplitNode(config=config)
    item_name_recognition_node = ItemNameRecognitionNode(config=config)
    embedding_node = BgeEmbeddingChunksNode(config=config)
    milvus_node = MilvusImportNode(config=config)

    # 第二步：注册“名字 → 可调用节点”的映射；注册先后顺序并不决定执行顺序。
    def tracked(node):
        """包装统一调用边界：先报告运行，再执行节点，最后报告成功或失败。"""
        if observer is None:
            return node

        def execute(state):
            """节点异常仍向图传播，观察者只记录进度，不替代原有异常处理。"""
            # 【节点进度顺序】running → BaseNode.__call__ → 节点 process → completed。
            # process 抛异常则报告 failed 并向门面传播；节点异常后不再执行后续图节点。
            observer(node.name, "running")
            try:
                update = node(state)
            except Exception:
                observer(node.name, "failed")
                raise
            observer(node.name, "completed")
            return update
        return execute

    workflow = StateGraph(ImportGraphState)
    workflow.add_node(entry_node.name, tracked(entry_node))
    workflow.add_node(pdf_to_md_node.name, tracked(pdf_to_md_node))
    workflow.add_node(md_to_img_node.name, tracked(md_to_img_node))
    workflow.add_node(document_split_node.name, tracked(document_split_node))
    workflow.add_node(item_name_recognition_node.name, tracked(item_name_recognition_node))
    workflow.add_node(embedding_node.name, tracked(embedding_node))
    workflow.add_node(milvus_node.name, tracked(milvus_node))

    # 第三步：用边定义真正的执行顺序。只有入口按文件格式分支，后续全部串行。
    workflow.add_edge(START, entry_node.name)
    workflow.add_conditional_edges(
        entry_node.name,
        _route_import_file,
        {
            "pdf": pdf_to_md_node.name,
            "markdown": md_to_img_node.name,
        },
    )
    # PDF 转换会在 state 中写入 md_path，之后与直接导入 Markdown 共用图片处理节点。
    workflow.add_edge(pdf_to_md_node.name, md_to_img_node.name)
    # 两条入口路径在图片处理后汇合，统一切分成可供后续 embedding 使用的 chunks。
    workflow.add_edge(md_to_img_node.name, document_split_node.name)
    # 商品名节点依赖已经完成的 chunks，并在同一个节点内完成识别、向量化、入库和回填。
    # 不拆成多条图边，可避免断点恢复时暴露“已识别但尚未写入 Milvus”的半完成状态。
    workflow.add_edge(document_split_node.name, item_name_recognition_node.name)
    # 商品名已回填到 chunks，编码服务才能把商品名和 content 组成最终编码文本。
    workflow.add_edge(item_name_recognition_node.name, embedding_node.name)
    # 全部切片编码通过后才进入写库，避免编码失败时已写入部分切片。
    workflow.add_edge(embedding_node.name, milvus_node.name)
    # Milvus 节点读回核验通过才设置 succeeded，END 本身不承担数据库验证。
    workflow.add_edge(milvus_node.name, END)

    logger.info(
        "导入流程图编排完成: entry=%s, pdf_node=%s, markdown_node=%s, "
        "split_node=%s, item_name_node=%s",
        entry_node.name,
        pdf_to_md_node.name,
        md_to_img_node.name,
        document_split_node.name,
        item_name_recognition_node.name,
    )
    return workflow.compile(name="import_graph")


def run_import_graph(
    state: ImportGraphState,
    config: ImportConfig | None = None,
    observer=None,
    source_archive=None,
) -> ImportGraphState:
    """对外执行入口：接收文件路径及可选任务/文档 ID，返回完整最终状态。

    task_id 用于追踪一次任务；document_id 用于定位可重复更新的业务文档。
    此入口每次从头运行，不从调用方提供的 chunks 或成功标志恢复。
    调用失败直接向上抛异常，调用方可保留 document_id 后重试完整导入。
    """
    # 【流程 06 · 图入口】由 ImportTaskFacade._run 调用；先创建独立状态，再取得串行锁并 invoke。
    # 图按下方流程 07.1～07.7 执行；核验最终结果后返回门面，继续流程 08 的产物归档。
    # 第一步：拒绝错误配置，再创建本次任务独立的初始状态。
    if not isinstance(state, dict):
        raise StateFieldError(field_name="state", expected_type=dict)
    config = config or get_config()
    validate_config(config)
    # 完整重跑只接收入口字段，防止上次成功态、主键或旧向量污染新任务。
    initial = create_default_state(**{key: state[key] for key in
                                   ("import_file_path", "document_id", "task_id") if key in state})
    # 归档地址只由任务门面传入，不接受 HTTP 请求任意指定内部存储路径。
    if source_archive is not None:
        initial["source_archive"] = source_archive
    initial["task_id"] = initial.get("task_id") or uuid.uuid4().hex
    initial["import_status"] = "running"
    task_id = initial["task_id"]
    logger.info("开始运行导入流程: task_id=%s", task_id)
    # 单进程只允许一个完整导入，图内串行本身不能保护两个并发 invoke。
    with _import_lock:
        # 锁只保护当前 Python 进程；多进程部署仍需在外层实现文档级协调。
        result = build_import_graph(config=config, observer=observer).invoke(initial)
    # 第三步：检查图的最终业务结果，不能仅因 invoke 没抛异常就判定导入成功。
    if (result.get("import_status") != "succeeded"
            or result.get("written_chunk_count") != len(result.get("chunks", []))):
        raise StateFieldError(message="导入未达到完整入库成功状态")
    logger.info("导入流程运行完成: task_id=%s", task_id)
    return result
