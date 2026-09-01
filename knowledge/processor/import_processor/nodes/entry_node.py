"""导入流程入口节点。"""

from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNode(BaseNode):
    """识别导入文件类型，并为 LangGraph 后续路由设置状态。"""

    name = "entry_node"
    _markdown_extensions = frozenset({".md", ".markdown"})

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """根据导入文件扩展名启用 PDF 或 Markdown 处理分支。"""
        if not isinstance(state, dict):
            raise StateFieldError(
                node_name=self.name,
                field_name="state",
                expected_type=dict,
            )

        import_file_path = state.get("import_file_path")
        if not isinstance(import_file_path, str) or not import_file_path.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
            )

        normalized_path = import_file_path.strip()
        extension = Path(normalized_path).suffix.casefold()
        is_pdf = extension == ".pdf"
        is_markdown = extension in self._markdown_extensions
        if not is_pdf and not is_markdown:
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
                message=(
                    "状态字段 'import_file_path' 仅支持 PDF 或 Markdown 文件，"
                    f"实际路径: {normalized_path}"
                ),
            )

        # 两个标志是互斥的入口路由信号；PDF 转换完成后再顺序进入 Markdown 节点。
        state["is_pdf_read_enabled"] = is_pdf
        state["is_md_read_enabled"] = is_markdown

        # Markdown 分支会直接进入 MdToImgNode，因此入口处需补齐其必需路径。
        if is_pdf:
            state["pdf_path"] = normalized_path
        else:
            state["md_path"] = normalized_path

        self.logger.info(
            "导入文件类型识别完成: file_type=%s, "
            "is_pdf_read_enabled=%s, is_md_read_enabled=%s",
            "pdf" if is_pdf else "markdown",
            is_pdf,
            is_markdown,
        )
        return state
