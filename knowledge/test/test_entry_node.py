"""EntryNode 的文档标识测试。"""

import hashlib
import tempfile
import unittest
from pathlib import Path

from knowledge.processor.import_processor.nodes.entry_node import EntryNode


class EntryNodeTest(unittest.TestCase):
    def test_missing_document_id_uses_file_content_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "manual.md"
            content = "# RS-12\n数字万用表".encode("utf-8")
            file_path.write_bytes(content)

            result = EntryNode().process({"import_file_path": str(file_path)})

        self.assertEqual(result["document_id"], hashlib.sha256(content).hexdigest())
        self.assertEqual(result["md_path"], str(file_path.resolve()))

    def test_explicit_document_id_has_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "manual.pdf"
            file_path.write_bytes(b"fake-pdf")

            result = EntryNode().process(
                {
                    "import_file_path": str(file_path),
                    "document_id": " business-document-id ",
                }
            )

        self.assertEqual(result["document_id"], "business-document-id")
        self.assertEqual(result["pdf_path"], str(file_path.resolve()))


if __name__ == "__main__":
    unittest.main()
