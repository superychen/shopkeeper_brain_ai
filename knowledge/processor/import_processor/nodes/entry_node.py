"""导入流程入口节点。"""

import hashlib
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
        input_path = Path(normalized_path).expanduser()
        extension = input_path.suffix.casefold()
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

        if not input_path.is_file():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
                message=f"导入文件不存在或不是普通文件: {input_path}",
            )

        input_path = input_path.resolve()
        normalized_path = str(input_path)
        document_id = state.get("document_id")
        if document_id is None or document_id == "":
            # 当前没有独立文档表时，以文件内容摘要提供跨重试稳定的幂等标识。
            # 例如同一个 RS-12.pdf 重试两次会得到相同 document_id，从而 upsert 同一条
            # Milvus 记录；即使文件改名，只要内容不变，标识也保持不变。
            state["document_id"] = self._hash_file(input_path)
        elif not isinstance(document_id, str) or not document_id.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="document_id",
                expected_type=str,
            )
        else:
            state["document_id"] = document_id.strip()

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

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """流式计算文件摘要，避免大 PDF 一次性读入内存。

        ``iter(callable, sentinel)`` 会反复调用 read，直到返回空字节；相当于 Java 中
        常见的 read-buffer 循环，但不需要手工维护 while 条件。
        """
        digest = hashlib.sha256()
        with file_path.open("rb") as source_file:
            for block in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
