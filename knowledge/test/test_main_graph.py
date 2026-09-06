"""导入主图的节点拓扑和 PDF/Markdown 执行顺序测试。"""

import unittest
from unittest.mock import patch

from knowledge.processor.import_processor.main_graph import build_import_graph
from knowledge.processor.import_processor.nodes.document_split_node import (
    DocumentSplitNode,
)
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import (
    ItemNameRecognitionNode,
)
from knowledge.processor.import_processor.nodes.md_to_img_node import MdToImgNode
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState


class MainGraphTest(unittest.TestCase):
    """确保两种入口最终都经过切分和商品名称节点。"""

    def test_graph_places_document_split_after_markdown_image_node(self) -> None:
        graph = build_import_graph().get_graph()
        edges = {(edge.source, edge.target) for edge in graph.edges}

        self.assertIn("document_split_node", graph.nodes)
        self.assertIn("item_name_recognition_node", graph.nodes)
        self.assertIn(("md_to_img_node", "document_split_node"), edges)
        self.assertIn(
            ("document_split_node", "item_name_recognition_node"),
            edges,
        )
        self.assertIn(("item_name_recognition_node", "__end__"), edges)
        self.assertNotIn(("document_split_node", "__end__"), edges)
        self.assertNotIn(("md_to_img_node", "__end__"), edges)

    def test_markdown_route_executes_entry_image_and_split_in_order(self) -> None:
        result = self._invoke_with_fake_nodes("manual.md")

        self.assertEqual(
            result["item_name"],
            "entry>md_to_img>document_split>item_name_recognition",
        )
        self.assertEqual(result["chunks"], [{"content": "切分完成"}])

    def test_pdf_route_executes_pdf_image_and_split_in_order(self) -> None:
        result = self._invoke_with_fake_nodes("manual.pdf")

        self.assertEqual(
            result["item_name"],
            "entry>pdf_to_md>md_to_img>document_split>item_name_recognition",
        )
        self.assertEqual(result["chunks"], [{"content": "切分完成"}])

    @staticmethod
    def _invoke_with_fake_nodes(import_file_path: str) -> ImportGraphState:
        """替换有文件/网络副作用的节点，只验证 LangGraph 的真实路由顺序。"""

        def append_trace(state: ImportGraphState, node_name: str) -> None:
            previous = state.get("item_name", "")
            state["item_name"] = f"{previous}>{node_name}" if previous else node_name

        def fake_entry(
            _node: EntryNode,
            state: ImportGraphState,
        ) -> ImportGraphState:
            append_trace(state, "entry")
            is_pdf = state["import_file_path"].casefold().endswith(".pdf")
            state["is_pdf_read_enabled"] = is_pdf
            state["is_md_read_enabled"] = not is_pdf
            return state

        def fake_pdf(
            _node: PdfToMdNode,
            state: ImportGraphState,
        ) -> ImportGraphState:
            append_trace(state, "pdf_to_md")
            state["md_path"] = "manual.md"
            return state

        def fake_image(
            _node: MdToImgNode,
            state: ImportGraphState,
        ) -> ImportGraphState:
            append_trace(state, "md_to_img")
            state["md_path"] = state.get("md_path", state["import_file_path"])
            state["md_content"] = "# 示例\n正文"
            return state

        def fake_split(
            _node: DocumentSplitNode,
            state: ImportGraphState,
        ) -> ImportGraphState:
            append_trace(state, "document_split")
            # 本测试只关心图编排；切分算法由 test_document_split_node 单独覆盖。
            state["chunks"] = [{"content": "切分完成"}]  # type: ignore[list-item]
            return state

        def fake_item_name(
            _node: ItemNameRecognitionNode,
            state: ImportGraphState,
        ) -> ImportGraphState:
            append_trace(state, "item_name_recognition")
            return state

        # patch.object 类似 Java 测试里的替身对象：保留真实图，只隔离节点内部副作用。
        with (
            patch.object(EntryNode, "process", fake_entry),
            patch.object(PdfToMdNode, "process", fake_pdf),
            patch.object(MdToImgNode, "process", fake_image),
            patch.object(DocumentSplitNode, "process", fake_split),
            patch.object(ItemNameRecognitionNode, "process", fake_item_name),
        ):
            return build_import_graph().invoke(
                {
                    "import_file_path": import_file_path,
                    "item_name": "",
                    "chunks": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
