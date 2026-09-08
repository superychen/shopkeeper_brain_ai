import hashlib
import os
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from pymilvus import MilvusClient

from knowledge.api.main import create_app
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.service.import_task_facade import ImportTaskFacade
from knowledge.utils.client.storage_clients import StorageClients


@unittest.skipUnless(os.getenv("RUN_API_IMPORT_INTEGRATION") == "1", "需要真实模型、MinIO 与 Milvus")
class RealImportApiTests(unittest.TestCase):
    def test_pdf_upload_to_both_stores(self):
        suffix = uuid.uuid4().hex
        bucket = "codex-it-api-" + suffix
        config = ImportConfig(chunks_collection="it_api_chunks_" + suffix,
                              item_name_collection="it_api_items_" + suffix, minio_bucket=bucket)
        facade = ImportTaskFacade(config)
        milvus = MilvusClient(uri=config.milvus_url, token=config.milvus_token or "", timeout=10)
        try:
            with tempfile.TemporaryDirectory() as directory:
                with pymupdf.open() as pdf:
                    page = pdf.new_page()
                    page.insert_text((72, 72), "RS PRO RS-12 Digital Multimeter", fontsize=18)
                    page.insert_text((72, 110), "Brand: RS PRO. Model: RS-12. Product: digital multimeter.")
                    original = pdf.tobytes()
                app = create_app(lambda: facade, Path(directory) / "runtime")
                with TestClient(app) as client:
                    response = client.post("/upload", files={"file": ("RS-12.pdf", original, "application/pdf")})
                    self.assertEqual(response.status_code, 202, response.text)
                    task = response.json()["task_id"]
                    deadline = time.monotonic() + 600
                    max_poll = 0
                    while time.monotonic() < deadline:
                        started = time.monotonic()
                        state = client.get("/status/" + task).json()
                        max_poll = max(max_poll, time.monotonic() - started)
                        if state["status"] in {"completed", "failed"}:
                            break
                        time.sleep(.5)
                    self.assertEqual(state["status"], "completed", state)
                    self.assertEqual(state["finished_steps"], 10)
                    archive = state["result"]["archive"]
                    minio = StorageClients.get_minio(config)
                    for item in archive["objects"]:
                        stream = minio.get_object(bucket, item["object_key"])
                        try:
                            payload = stream.read()
                        finally:
                            stream.close()
                            stream.release_conn()
                        self.assertEqual(hashlib.sha256(payload).hexdigest(), item["sha256"])
                        if item["role"] == "original":
                            self.assertEqual(payload, original)
                    rows = milvus.query(collection_name=config.chunks_collection, filter="",
                                        output_fields=["metadata"], limit=1, consistency_level="Strong")
                    self.assertEqual(rows[0]["metadata"]["source_archive"]["task_id"], task)
                    self.assertFalse((Path(directory) / "runtime" / task).exists())
                    print("API PDF integration verified; max polling latency:", round(max_poll, 3), "seconds")
        finally:
            facade.close()
            for name in (config.chunks_collection, config.item_name_collection):
                if name.startswith("it_api_") and name.endswith(suffix) and milvus.has_collection(name):
                    milvus.drop_collection(name)
            milvus.close()
            minio = StorageClients.get_minio(config)
            if bucket == "codex-it-api-" + suffix and minio.bucket_exists(bucket):
                for obj in minio.list_objects(bucket, recursive=True):
                    minio.remove_object(bucket, obj.object_name)
                minio.remove_bucket(bucket)


if __name__ == "__main__":
    unittest.main()
