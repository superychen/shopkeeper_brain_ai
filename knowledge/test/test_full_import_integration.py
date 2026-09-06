"""真实 DeepSeek、BGE-M3 和 Milvus 的完整入口验收，默认不运行。"""

import os
import tempfile
import unittest
import uuid
from pathlib import Path

from pymilvus import MilvusClient
import pymupdf as fitz

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.main_graph import run_import_graph
from knowledge.utils.client.storage_clients import StorageClients


@unittest.skipUnless(os.getenv("RUN_FULL_IMPORT_INTEGRATION") == "1", "完整验收会调用 DeepSeek 并加载 BGE-M3")
class FullImportIntegrationTest(unittest.TestCase):
    def test_real_pdf_entry(self):
        suffix = uuid.uuid4().hex
        config = ImportConfig(chunks_collection="it_pdf_chunks_" + suffix,
                              item_name_collection="it_pdf_items_" + suffix)
        client = MilvusClient(uri=config.milvus_url, token=config.milvus_token or "", timeout=10)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "RS-12.pdf"
                with fitz.open() as pdf:
                    page = pdf.new_page()
                    page.insert_text((72, 72), "RS PRO RS-12 Digital Multimeter", fontsize=18)
                    page.insert_text((72, 110), "The product is RS PRO RS-12 digital multimeter.\nBrand: RS PRO. Model: RS-12.", fontsize=12)
                    pdf.save(path)
                result = run_import_graph({"import_file_path": str(path)}, config)
            self.assertEqual(result["import_status"], "succeeded")
            self.assertGreater(result["written_chunk_count"], 0)
            self.assertTrue(result["is_pdf_read_enabled"])
            self.assertEqual(len(result["chunks"][0]["dense_vector"]), 1024)
            print("PDF import verified:", result["written_chunk_count"], "chunks")
        finally:
            for name in (config.chunks_collection, config.item_name_collection):
                if client.has_collection(collection_name=name, timeout=10):
                    client.drop_collection(collection_name=name, timeout=10)
            client.close()

    def test_real_markdown_entry_through_all_models_and_milvus(self):
        suffix = uuid.uuid4().hex
        test_bucket = "codex-it-full-" + suffix
        config = ImportConfig(chunks_collection="it_full_chunks_" + suffix,
                              item_name_collection="it_full_items_" + suffix,
                              minio_bucket=test_bucket)
        client = MilvusClient(uri=config.milvus_url, token=config.milvus_token or "", timeout=10)
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "RS PRO RS-12 数字万用表.md"
                images = Path(folder) / "images"
                images.mkdir()
                with fitz.open() as illustration:
                    page = illustration.new_page(width=400, height=200)
                    page.insert_text((20, 40), "RS PRO RS-12", fontsize=20)
                    page.insert_text((20, 80), "Digital Multimeter", fontsize=16)
                    page.get_pixmap().save(images / "product.png")
                path.write_text("# RS PRO RS-12 数字万用表\n"
                                "本说明书的唯一产品为 RS PRO RS-12 数字万用表。\n"
                                "## 产品资料\n品牌为 RS PRO，型号为 RS-12，产品类型为数字万用表。\n"
                                '<img src="images/product.png">\n',
                                encoding="utf-8")
                first = run_import_graph({"import_file_path": str(path)}, config)
                second = run_import_graph({"import_file_path": str(path)}, config)
            self.assertEqual(first["item_name_status"], "recognized")
            self.assertEqual(first["import_status"], "succeeded")
            self.assertEqual(first["milvus_chunk_ids"], second["milvus_chunk_ids"])
            self.assertEqual(len(first["chunks"][0]["dense_vector"]), 1024)
            self.assertTrue(first["chunks"][0]["sparse_vector"])
            self.assertEqual(first["image_summary_fallback_count"], 0)
            self.assertEqual(first["image_upload_failure_count"], 0)
            self.assertIn(config.minio_bucket, first["md_content"])
            print("Full import verified:", first["written_chunk_count"], "chunks; dense_dim=1024; repeated IDs stable")
        finally:
            try:
                storage = StorageClients._minio_client
                # 清理目标由本次UUID直接构造，绝不从环境配置或默认业务bucket取值。
                if test_bucket != "codex-it-full-" + suffix or config.minio_bucket != test_bucket:
                    raise RuntimeError("Refusing cleanup outside the generated test bucket")
                if storage is not None and storage.bucket_exists(test_bucket):
                    storage.remove_object(test_bucket, "knowledge/RS PRO RS-12 数字万用表/product.png")
                    storage.remove_bucket(test_bucket)
            finally:
                # 仅删除本次 UUID 隔离的两个集合，保留用户已有知识库。
                for name in (config.chunks_collection, config.item_name_collection):
                    if client.has_collection(collection_name=name, timeout=10):
                        client.drop_collection(collection_name=name, timeout=10)
                client.close()


if __name__ == "__main__":
    unittest.main()
