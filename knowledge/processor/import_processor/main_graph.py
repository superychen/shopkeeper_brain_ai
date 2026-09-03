"""导入流程 LangGraph 编排。"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.nodes.document_split_node import (
    DocumentSplitNode,
)
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.md_to_img_node import MdToImgNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState

logger = logging.getLogger("import.main_graph")

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
) -> CompiledStateGraph:
    """创建并编译 PDF/Markdown 导入流程图。"""
    entry_node = EntryNode(config=config)
    pdf_to_md_node = PdfToMdNode(config=config)
    md_to_img_node = MdToImgNode(config=config)
    document_split_node = DocumentSplitNode(config=config)

    workflow = StateGraph(ImportGraphState)
    workflow.add_node(entry_node.name, entry_node)
    workflow.add_node(pdf_to_md_node.name, pdf_to_md_node)
    workflow.add_node(md_to_img_node.name, md_to_img_node)
    workflow.add_node(document_split_node.name, document_split_node)

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
    # Embedding/Milvus 节点接入前，文档切分节点暂时作为导入图的末端。
    workflow.add_edge(document_split_node.name, END)

    logger.info(
        "导入流程图编排完成: entry=%s, pdf_node=%s, markdown_node=%s, "
        "split_node=%s",
        entry_node.name,
        pdf_to_md_node.name,
        md_to_img_node.name,
        document_split_node.name,
    )
    return workflow.compile(name="import_graph")


def run_import_graph(
    state: ImportGraphState,
    config: ImportConfig | None = None,
) -> ImportGraphState:
    """运行一次导入流程并返回最终状态。"""
    task_id = state.get("task_id", "") if isinstance(state, dict) else ""
    logger.info("开始运行导入流程: task_id=%s", task_id)
    result = build_import_graph(config=config).invoke(state)
    logger.info("导入流程运行完成: task_id=%s", task_id)
    return result
