"""LangGraph 节点之间传递的数据契约，可视为整个流程共享的“任务数据包”。

Entry 写路径 → PDF/图片节点写 md_content → 切分节点写 chunks → 商品名节点回填名称
→ 编码节点写向量与模型指纹 → 入库节点写主键、数量和 succeeded。
TypedDict 只提供类型提示，运行时仍是普通 dict；实际校验由节点和校验服务完成。
"""

import copy
from typing import Literal, NotRequired, TypedDict


class ChunkRecord(TypedDict):
    """Step 4 输出给向量化节点的稳定切片结构。

    ``content`` 是后续 embedding 的主文本；其余字段保留标题路径和原始 section
    映射，便于检索结果展示、问题排查以及未来按章节过滤。
    """

    chunk_index: int  # 从 0 开始连续编号，参与生成稳定的 Milvus 主键
    file_title: str  # 文档标题，提供文档级上下文
    title: str  # 当前切片标题
    parent_title: str  # 父级标题；没有父级时允许空字符串
    heading_path: list[str]  # 从上级到当前章节的标题路径，用于溯源
    source_titles: list[str]  # 合并进此切片的原始章节标题
    source_section_indexes: list[int]  # 原始章节位置，便于追踪合并/拆分结果
    body: str  # 切片正文，不含额外组装的标题上下文
    content: str  # 组装后的检索文本，编码服务在此基础上加入商品名
    char_count: int  # content 的字符数，不能用来替代数据库字节大小检查
    source_path: str  # 来源文件路径，用于检索结果追溯

    # 切分阶段不生成商品名，识别成功后才由商品名称节点补充。
    item_name: NotRequired[str]
    # NotRequired 表示该字段在前面的节点尚不存在，不代表字段值可以是 None。
    dense_vector: NotRequired[list[float]]  # 编码节点生成，表达语义相似度
    sparse_vector: NotRequired[dict[int, float]]  # 编码节点生成，token ID → 正权重
    embedding_text_hash: NotRequired[str]  # 编码文本摘要，写库前核对文本是否被改动
    record_hash: NotRequired[str]  # 入库节点回填，用于核验记录内容及编码配置
    chunk_id: NotRequired[str]  # 入库成功回填，重跑同文档同位置时保持稳定

class ImportGraphState(TypedDict, total=False):
    """整个任务的状态；total=False 允许节点只返回本次更新的部分字段。"""

    # ==================== 任务标识 ====================

    task_id: str  # 任务 ID，用于任务追踪(web交互的时候用到，实时看到节点的处理日志)
    document_id: str  # 跨重试稳定的文档标识，用于 Milvus 幂等写入

    # ==================== 控制标志 ====================

    is_md_read_enabled: bool  # 是否启用 MD 读取

    is_pdf_read_enabled: bool  # 是否启用 PDF 读取

    # ==================== 路径信息 ====================

    import_file_path: str  # 导入文件路径

    file_dir: str  # 导入(出)文件目录

    pdf_path: str  # PDF 文件路径

    md_path: str  # 转换后Markdown 文件路径

    # ==================== 文件信息 ====================

    file_title: str  # 文件标题（不含扩展名）

    item_name: str  # 识别出的商品/产品名称(方便程序员用)
    item_name_status: Literal["recognized", "not_found", "ambiguous"]
    item_name_confidence: float  # 仅表示输入证据充分度，不代表商品质量评分
    item_name_evidence: list[str]  # 例如 ["RS PRO", "RS-12", "数字万用表"]
    item_name_milvus_pk: str  # 写库成功后回填的稳定 SHA-256 主键

    # ==================== 处理中间数据 ====================

    # API 门面给出归档对象键；raw_md_path 在图片处理前保存，避免丢失原始转换内容。
    source_archive: dict
    raw_md_path: str
    md_content: str  # Markdown 文档内容

    chunks: list[ChunkRecord]  # Step 4 组装完成、可直接交给向量化节点的切片
    import_status: Literal["pending", "running", "succeeded"]  # 失败以异常退出，仅最终入库节点标记成功
    embedding_model: str  # 编码节点记录的模型名称
    embedding_profile: str  # 模型/分词器/模板指纹，防止同集合混入不兼容向量
    split_profile: str  # 切分配置指纹，用于追踪切片生成方式
    embedded_chunk_count: int  # 已成功生成两种向量的切片数
    chunks_collection: str  # 实际写入的切片集合名
    milvus_chunk_ids: list[str]  # 与 chunks 顺序对应的持久化主键
    written_chunk_count: int  # 核验通过的写入条数，包含覆盖更新
    image_summary_fallback_count: int  # 图片无法生成模型摘要时的降级计数
    image_upload_failure_count: int  # 图片上传失败计数，供结果摘要展示
    import_warnings: list[str]  # 可以继续导入但需要调用方关注的降级说明

# 这是默认模板，不能直接作为运行状态使用，内部的列表属于可变对象。
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "document_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    "item_name_status": "not_found",
    "item_name_confidence": 0.0,
    "item_name_evidence": [],
    "item_name_milvus_pk": "",
    "import_status": "pending",
    "embedding_model": "",
    "embedding_profile": "",
    "split_profile": "",
    "embedded_chunk_count": 0,
    "chunks_collection": "",
    "milvus_chunk_ids": [],
    "written_chunk_count": 0,
    "image_summary_fallback_count": 0,
    "image_upload_failure_count": 0,
    "import_warnings": [],
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", import_file_path="doc.pdf")
    """
    # 深复制使 chunks、warnings 等列表相互独立，避免多个任务污染默认模板。
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """
    获取默认状态副本

    Returns:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
