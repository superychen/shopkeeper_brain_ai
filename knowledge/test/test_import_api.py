import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from knowledge.api.main import create_app
from knowledge.api.dependencies import get_import_service
from knowledge.service.import_service import ImportService
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.service.import_task_facade import ImportTaskFacade


class Artifacts:
    def put(self, path, key, role):
        return {"object_key": key, "role": role}

    def finish(self, context, state, objects):
        return {"object_key": context["manifest"]}


class ImportApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.release = threading.Event()
        self.entered = threading.Event()

        def runner(state, *, config, observer, source_archive):
            observer("entry_node", "running")
            self.entered.set()
            self.release.wait(5)
            observer("entry_node", "completed")
            for name in ("md_to_img_node", "document_split_node", "item_name_recognition_node",
                         "bge_embedding_chunks_node", "milvus_import_node"):
                observer(name, "running")
                observer(name, "completed")
            return dict(written_chunk_count=2, chunks_collection="test", embedding_model="test")

        self.facade = ImportTaskFacade(ImportConfig(), capacity=1, runner=runner, artifacts=Artifacts())
        self.app = create_app(lambda: self.facade, self.temp.name, max_file_bytes=1024)
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.release.set()
        self.client.__exit__(None, None, None)
        self.assertEqual(self.app.state.import_service_provider.cache_info().currsize, 0)
        self.temp.cleanup()

    def upload(self, content=b"# Manual\nhello"):
        return self.client.post("/upload", files={"file": ("manual.md", content)})

    def test_service_factory_is_cached_across_requests(self):
        provider = self.app.state.import_service_provider
        first = provider()
        self.client.get("/status/missing-one")
        self.client.get("/status/missing-two")
        self.assertIs(first, provider())
        self.assertEqual(provider.cache_info().misses, 1)
        self.assertGreaterEqual(provider.cache_info().hits, 4)

    def test_route_service_can_be_overridden_by_depends(self):
        service = Mock(spec=ImportService)
        service.upload.return_value = dict(task_id="injected-task", document_id="doc-1",
                                           status="queued", status_url="/status/injected-task",
                                           poll_interval_ms=1500, total_steps=9)
        self.app.dependency_overrides[get_import_service] = lambda: service
        try:
            response = self.upload()
            self.assertEqual(response.status_code, 202)
            self.assertEqual(response.json()["task_id"], "injected-task")
            service.upload.assert_called_once()
            self.assertEqual(service.upload.call_args.args[0], "manual.md")
            self.assertFalse(self.facade.store.records)
        finally:
            self.app.dependency_overrides.clear()

    def wait_terminal(self, task):
        for _ in range(100):
            data = self.client.get("/status/" + task).json()
            if data["status"] in {"completed", "failed"}:
                return data
            time.sleep(.02)
        self.fail("task did not finish")

    def test_poll_during_blocking_import_and_queue_limit(self):
        first = self.upload()
        self.assertEqual(first.status_code, 202)
        self.assertTrue(self.entered.wait(2))
        task = first.json()["task_id"]
        response = self.client.get("/status/" + task)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.json()["running_list"], ["entry_node"])
        second = self.upload()
        self.assertEqual(second.status_code, 202)
        self.assertEqual(self.client.get("/status/" + second.json()["task_id"]).json()["status"], "queued")
        self.assertEqual(self.upload().status_code, 429)
        self.release.set()
        result = self.wait_terminal(task)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["total_steps"], 9)
        self.assertEqual(result["progress_percent"], 100)
        self.assertEqual(result["running_list"], [])
        self.assertEqual(result["result"]["archive_status"], "verified")
        self.wait_terminal(second.json()["task_id"])

    def test_archive_failure_is_not_success(self):
        def fail(*args):
            raise RuntimeError("private details")
        self.facade.artifacts.finish = fail
        self.release.set()
        task = self.upload().json()["task_id"]
        result = self.wait_terminal(task)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["result"]["vector_status"], "verified")
        self.assertEqual(result["error"]["step_id"], "backup_artifacts")
        self.assertNotIn("private details", str(result))
        self.assertTrue((Path(self.temp.name) / task / "original/manual.md").exists())

    def test_upload_validation_and_cors(self):
        self.assertEqual(self.upload(b"").status_code, 422)
        self.assertEqual(self.upload(b"a" * 1025).status_code, 413)
        self.assertEqual(self.upload(b"\xff").status_code, 422)
        self.assertEqual(self.client.post("/upload", files={"file": ("../bad.md", b"hi")}).status_code, 422)
        self.assertEqual(self.client.post("/upload", files={"file": ("bad.exe", b"hi")}).status_code, 415)
        self.assertEqual(self.client.get("/status/missing").status_code, 404)
        response = self.client.options("/upload", headers={"Origin": "http://localhost:5500", "Access-Control-Request-Method": "POST"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5500")
        self.assertEqual(self.client.options("/upload", headers={"Origin": "http://bad.invalid", "Access-Control-Request-Method": "POST"}).status_code, 400)
        self.assertEqual(self.client.get("/import").status_code, 200)

    def test_pydantic_parameters_fail_before_queue_and_disk(self):
        response = self.client.post("/upload", files={"file": ("manual.md", b"hello")},
                                    data={"document_id": "中" * 43})
        self.assertEqual(response.status_code, 422)
        self.assertIn("document_id", response.json()["detail"])
        self.assertFalse(self.facade.store.records)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        # 验证失败不能泄漏队列名额，否则后续合法上传会错误返回 429。
        self.release.set()
        response = self.client.post("/upload", files={"file": ("manual.md", b"hello")},
                                    data={"document_id": "  doc-1  "})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["document_id"], "doc-1")
        self.wait_terminal(response.json()["task_id"])

    def test_original_backup_failure_stops_graph(self):
        def fail(*args):
            raise RuntimeError("offline")
        self.facade.artifacts.put = fail
        task = self.upload().json()["task_id"]
        result = self.wait_terminal(task)
        self.assertFalse(self.entered.is_set())
        self.assertEqual(result["error"]["step_id"], "backup_original")
        self.assertEqual(result["running_list"], [])
        self.assertEqual(result["result"]["vector_status"], "not_started")

    def test_chunked_request_body_limit(self):
        boundary = b"example-boundary"
        def chunks():
            yield b'--' + boundary + b'\r\nContent-Disposition: form-data; name="file"; filename="x.md"\r\n\r\n'
            for _ in range(18):
                yield b"x" * 65536
            yield b'\r\n--' + boundary + b'--\r\n'
        response = self.client.post("/upload", content=chunks(), headers={
            "Content-Type": "multipart/form-data; boundary=example-boundary",
            "Origin": "http://localhost:5500",
        })
        self.assertEqual(response.status_code, 413, response.text)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:5500")

    def test_work_copy_failure_marks_preparation_step_failed(self):
        with patch("knowledge.service.import_task_facade.shutil.copyfile", side_effect=OSError("copy failed")):
            task = self.upload().json()["task_id"]
            result = self.wait_terminal(task)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["step_id"], "backup_original")
        step = next(s for s in result["steps"] if s["id"] == "backup_original")
        self.assertEqual(step["status"], "failed")
        self.assertEqual(result["running_list"], [])
        self.assertFalse(self.entered.is_set())
        self.assertTrue((Path(self.temp.name) / task / "original/manual.md").is_file())

    def test_failure_is_visible_while_failure_archive_is_blocked(self):
        archive_entered = threading.Event()
        release_archive = threading.Event()

        def record_failure(*args):
            archive_entered.set()
            release_archive.wait(5)
            raise OSError("archive offline")

        self.facade.artifacts.put = Mock(side_effect=OSError("original offline"))
        self.facade.artifacts.record_failure = record_failure
        try:
            task = self.upload().json()["task_id"]
            self.assertTrue(archive_entered.wait(2))
            result = self.client.get("/status/" + task).json()
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["running_list"], [])
            self.assertEqual(result["error"]["step_id"], "backup_original")
            self.assertEqual(result["result"]["vector_status"], "not_started")
            self.assertFalse(self.entered.is_set())
        finally:
            release_archive.set()
            self.facade.queue.join()
        self.assertEqual(self.client.get("/status/" + task).json(), result)


if __name__ == "__main__":
    unittest.main()
