"""导入流程状态类型定义。"""

import copy
from typing import Literal, NotRequired, TypedDict


class ChunkRecord(TypedDict):
    """Step 4 输出给向量化节点的稳定切片结构。

    ``content`` 是后续 embedding 的主文本；其余字段保留标题路径和原始 section
    映射，便于检索结果展示、问题排查以及未来按章节过滤。
    """

    chunk_index: int
    file_title: str
    title: str
    parent_title: str
    heading_path: list[str]
    source_titles: list[str]
    source_section_indexes: list[int]
    body: str
    content: str
    char_count: int
    source_path: str

    # 切分阶段不生成商品名，识别成功后才由商品名称节点补充。
    item_name: NotRequired[str]

class ImportGraphState(TypedDict, total=False):
    """包含整个导入流程中传递的数据。"""

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

    md_content: str  # Markdown 文档内容

    chunks: list[ChunkRecord]  # Step 4 组装完成、可直接交给向量化节点的切片
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
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段

    Returns:
        新的状态实例

    Examples:
        >>> state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
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
