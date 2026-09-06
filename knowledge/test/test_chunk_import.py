"""验证批次故障、幂等覆盖和真实 LangGraph 串行执行，隔离外部模型与存储。"""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

import numpy as np
from scipy.sparse import csr_array
from pymilvus import MilvusClient

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import EmbeddingError, MilvusError, ConfigurationError, StateFieldError
from knowledge.processor.import_processor.main_graph import run_import_graph
from knowledge.processor.import_processor.main_graph import build_import_graph
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.md_to_img_node import MdToImgNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.bge_embedding_chunks_node import BgeEmbeddingChunksNode
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.processor.import_processor.nodes.item_name_recognition_node import ItemNameRecognitionNode
from knowledge.processor.import_processor.utils.markdown_image_util import normalize_html_images
from knowledge.service.chunk_repository import ChunkRepository
from knowledge.service.chunk_embedding_service import ChunkEmbeddingService
from knowledge.service.embedding_vector_util import extract_vectors, validate_vectors
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


def make_state(count=3):
    chunks = []
    for i in range(count):
        content = f"## 参数{i}\n交流电压测量范围"
        chunks.append(dict(chunk_index=i, file_title="RS-12", title=f"参数{i}", parent_title="说明",
                           heading_path=["说明"], source_titles=["参数"], source_section_indexes=[i],
                           content=content, body=content, char_count=len(content), source_path="manual.md"))
    return {"document_id": "doc-1", "item_name": "", "chunks": chunks}


class FakeTokenizer:
    model_max_length = 8192
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text) + 2))}


class FakeBge:
    def __init__(self, fail_at=None):
        self.model = SimpleNamespace(tokenizer=FakeTokenizer())
        self.calls = []
        self.fail_at = fail_at

    def encode_documents(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("injected encoding failure")
        return {"dense": [[0.1, 0.2, 0.3] for _ in texts],
                "sparse": csr_array([[0, 0.7, 0.3] for _ in texts])}


class MemoryMilvus:
    """以主键覆盖实体；可模拟响应丢失和错误回执，不假设批次事务。"""
    def __init__(self):
        self.schema = None
        self.indexes = None
        self.rows = {}
        self.writes = 0
        self.fail_at = None
        self.bad_count = False
        self.corrupt_read = False
        self.lose_response_once = False
        self.fail_delete = False

    def has_collection(self, **kwargs):
        return self.schema is not None

    def create_schema(self, **kwargs):
        return MilvusClient.create_schema(**kwargs)

    def prepare_index_params(self):
        return MilvusClient.prepare_index_params()

    def create_collection(self, schema, index_params, **kwargs):
        self.schema, self.indexes = schema, index_params

    def describe_collection(self, **kwargs):
        return self.schema.to_dict()

    def list_indexes(self, **kwargs):
        return [row.index_name for row in self.indexes]

    def describe_index(self, index_name, **kwargs):
        row = next(row.to_dict() for row in self.indexes if row.index_name == index_name)
        row["metric_type"] = row.get("metric_type", row.get("params", {}).get("metric_type"))
        return row

    def load_collection(self, **kwargs):
        pass

    def upsert(self, data, **kwargs):
        self.writes += 1
        if self.writes == self.fail_at:
            raise RuntimeError("injected write failure")
        self.rows.update({row["chunk_id"]: copy.deepcopy(row) for row in data})
        if self.lose_response_once:
            self.lose_response_once = False
            raise TimeoutError("lost acknowledgement")
        return {"upsert_count": len(data) - int(self.bad_count)}

    def get(self, ids, **kwargs):
        rows = [copy.deepcopy(self.rows[key]) for key in ids if key in self.rows]
        if self.corrupt_read and rows:
            rows[0]["record_hash"] = "wrong"
        return rows

    def query(self, filter, output_fields, **kwargs):
        if filter.startswith("embedding_profile != "):
            profile = json.loads(filter.split(" != ", 1)[1])
            return [row for row in self.rows.values() if row["embedding_profile"] != profile][:1]
        document_id = json.loads(filter.split(" == ", 1)[1])
        return [{"count(*)": sum(row["document_id"] == document_id for row in self.rows.values())}]

    def delete(self, filter, **kwargs):
        if self.fail_delete:
            raise RuntimeError("injected cleanup failure")
        doc, tail = filter.split(" and chunk_index >= ")
        document_id = json.loads(doc.split(" == ", 1)[1])
        self.rows = {key: row for key, row in self.rows.items()
                     if not (row["document_id"] == document_id and row["chunk_index"] >= int(tail))}


class ChunkImportTest(unittest.TestCase):
    def setUp(self):
        self.config = ImportConfig(embedding_dim=3, embedding_batch_size=8, milvus_batch_size=2,
                                   milvus_max_retries=0, chunks_collection="unit_chunks")

    def embed(self, state, model=None):
        with patch.object(AIClients, "get_bge_m3", return_value=model or FakeBge()), \
                patch.object(ChunkEmbeddingService, "_profile", return_value="profile-v1"):
            return {**state, **BgeEmbeddingChunksNode(self.config).process(state)}

    def test_batch_order_and_input_not_mutated(self):
        model, state = FakeBge(), make_state(20)
        original = copy.deepcopy(state)
        with patch.object(AIClients, "get_bge_m3", return_value=model) as get_model, \
                patch.object(ChunkEmbeddingService, "_profile", return_value="p"):
            result = BgeEmbeddingChunksNode(self.config).process(state)
        self.assertEqual([len(batch) for batch in model.calls], [8, 8, 4])
        get_model.assert_called_once()
        self.assertEqual(state, original)
        self.assertEqual([c["chunk_index"] for c in result["chunks"]], list(range(20)))
        self.assertEqual(result["chunks"][0]["sparse_vector"], {1: 0.7, 2: 0.3})

    def test_second_encoding_batch_failure_keeps_input_clean(self):
        state = make_state(9)
        before = copy.deepcopy(state)
        with self.assertRaises(EmbeddingError):
            self.embed(state, FakeBge(fail_at=2))
        self.assertEqual(state, before)

    def test_bad_inputs_fail_before_model(self):
        for bad in ([], [None], [{"chunk_index": True}]):
            with self.subTest(bad=bad), patch.object(AIClients, "get_bge_m3") as model:
                with self.assertRaises(StateFieldError):
                    BgeEmbeddingChunksNode(self.config).process({"document_id": "doc", "chunks": bad})
                model.assert_not_called()
        for value in (0, -1, True):
            self.config.embedding_batch_size = value
            with self.assertRaises(ConfigurationError):
                BgeEmbeddingChunksNode(self.config).process(make_state())

    def test_token_overflow_never_encodes(self):
        self.config.embedding_max_tokens = 3
        model = FakeBge()
        with self.assertRaises(EmbeddingError):
            self.embed(make_state(), model)
        self.assertFalse(model.calls)

    def test_invalid_vectors_and_csr(self):
        for dense, sparse in (([0, 0, 0], {1: 1}), ([1, float("nan"), 1], {1: 1}),
                              ([1, 1e40, 1], {1: 1}), ([1, 1, 1], {}),
                              ([1, 1, 1], {1: 1, "1": 2}), ([1, 1, 1], {1.2: 1}),
                              ([1, 1, 1], {True: 1}), ([1, 1, 1], {1: -1})):
            with self.subTest(dense=dense, sparse=sparse), self.assertRaises(EmbeddingError):
                validate_vectors(dense, sparse, 3)
        broken = SimpleNamespace(indptr=np.array([0, 2]), indices=np.array([1]), data=np.array([1]))
        with self.assertRaises(EmbeddingError):
            extract_vectors({"dense": [[1, 1, 1]], "sparse": broken}, 1, 3)
        with self.assertRaises(EmbeddingError):
            extract_vectors({"dense": [[1, 1, 1]], "sparse": []}, 1, 3)

    def test_upsert_reimport_shorter_and_document_isolation(self):
        db = MemoryMilvus()
        with patch.object(StorageClients, "get_milvus", return_value=db):
            node = MilvusImportNode(self.config)
            state = self.embed(make_state(5))
            first = node.process(state)
            second = node.process(state)
            self.assertEqual(first["milvus_chunk_ids"], second["milvus_chunk_ids"])
            other = self.embed({**make_state(2), "document_id": 'other"doc'})
            node.process(other)
            node.process(self.embed(make_state(3)))
        self.assertEqual(len(db.rows), 5)
        self.assertEqual(first["import_status"], "succeeded")
        self.assertNotIn("chunk_id", state["chunks"][0])

    def test_partial_write_retry_converges(self):
        db = MemoryMilvus()
        state = self.embed(make_state(5))
        before = copy.deepcopy(state)
        db.fail_at = 2
        with patch.object(StorageClients, "get_milvus", return_value=db):
            node = MilvusImportNode(self.config)
            with self.assertRaises(MilvusError):
                node.process(state)
            self.assertEqual(len(db.rows), 2)
            self.assertEqual(state, before)
            db.fail_at = None
            result = node.process(state)
        self.assertEqual(result["written_chunk_count"], 5)
        self.assertEqual(len(db.rows), 5)

    def test_lost_response_retries_same_primary_keys(self):
        self.config.milvus_max_retries = 1
        db = MemoryMilvus()
        db.lose_response_once = True
        with patch.object(StorageClients, "get_milvus", return_value=db), patch("knowledge.service.chunk_repository.time.sleep"):
            MilvusImportNode(self.config).process(self.embed(make_state(2)))
        self.assertEqual(db.writes, 2)
        self.assertEqual(len(db.rows), 2)

    def test_receipt_readback_and_cleanup_failures_are_not_success(self):
        state = self.embed(make_state())
        for fault in ("bad_count", "corrupt_read", "fail_delete"):
            db = MemoryMilvus()
            setattr(db, fault, True)
            with self.subTest(fault=fault), patch.object(StorageClients, "get_milvus", return_value=db):
                with self.assertRaises(MilvusError):
                    MilvusImportNode(self.config).process(state)
        self.assertNotEqual(state["import_status"], "succeeded")

    def test_schema_and_model_mismatch_rejected(self):
        db = MemoryMilvus()
        with patch.object(StorageClients, "get_milvus", return_value=db):
            state = self.embed(make_state())
            MilvusImportNode(self.config).process(state)
            with self.assertRaises(ConfigurationError):
                MilvusImportNode(self.config).process({**state, "embedding_profile": "other-model"})
            self.config.embedding_dim = 4
            with self.assertRaises(ConfigurationError):
                ChunkRepository(self.config).ensure_collection(db)

    def test_graph_starts_at_real_file_and_finishes_at_verified_database(self):
        db = MemoryMilvus()
        def no_item(_node, state):
            return {"item_name": "", "item_name_status": "not_found"}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "说明书.md"
            path.write_text("# 使用方法\n使用万用表测量交流电压。", encoding="utf-8")
            with patch.object(ItemNameRecognitionNode, "process", no_item), \
                    patch.object(AIClients, "get_bge_m3", return_value=FakeBge()), \
                    patch.object(ChunkEmbeddingService, "_profile", return_value="p"), \
                    patch.object(StorageClients, "get_milvus", return_value=db):
                result = run_import_graph({"import_file_path": str(path), "import_status": "succeeded",
                                           "milvus_chunk_ids": ["old"]}, self.config)
        self.assertEqual(result["import_status"], "succeeded")
        self.assertEqual(result["written_chunk_count"], len(db.rows))
        self.assertNotIn("old", result["milvus_chunk_ids"])

    def test_html_images_preserve_code(self):
        content = '<img src="images/接线 图.jpg">\n```html\n<img src="fake.jpg">\n```\n`<img src="inline.jpg">`'
        result = normalize_html_images(content)
        self.assertIn("%20", result)
        self.assertIn('<img src="fake.jpg">', result)
        self.assertIn('`<img src="inline.jpg">`', result)

    def test_changed_text_cannot_reuse_old_vectors(self):
        state = self.embed(make_state())
        state["chunks"][0]["content"] += "changed"
        state["chunks"][0]["char_count"] = len(state["chunks"][0]["content"])
        with patch.object(StorageClients, "get_milvus") as client, self.assertRaises(MilvusError):
            MilvusImportNode(self.config).process(state)
        client.assert_not_called()

    def test_html_image_upload_and_code_protection(self):
        config = ImportConfig(minio_bucket="test-bucket", minio_endpoint="127.0.0.1:9000")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "images" / "接线 图.jpg").write_bytes(b"fixture")
            path = root / "manual.md"
            code = '```html\n<img src="images/接线 图.jpg">\n```'
            path.write_text('# 图示\n<img src="images/接线 图.jpg">\n' + code, encoding="utf-8")
            node = MdToImgNode(config)
            node._image_uploader._client_factory = lambda: SimpleNamespace(fput_object=lambda **kwargs: None)
            image_path = (root / "images" / "接线 图.jpg").resolve()
            with patch.object(node._vlm_summarizer, "summarize_all", return_value={image_path: "接线说明"}):
                result = node.process({"md_path": str(path)})
            self.assertIn("http://127.0.0.1:9000/test-bucket", result["md_content"])
            self.assertIn(code, result["md_content"])
            self.assertEqual(result["image_upload_failure_count"], 0)

    def test_graph_failure_halts_all_downstream_nodes(self):
        stages = [EntryNode, MdToImgNode, DocumentSplitNode, ItemNameRecognitionNode,
                  BgeEmbeddingChunksNode, MilvusImportNode]
        for failed_index in (3, 4, 5):
            visited = []
            def handler(index):
                def invoke(_node, state):
                    visited.append(index)
                    if index == failed_index:
                        raise EmbeddingError(message="injected graph failure")
                    return {"is_md_read_enabled": True, "is_pdf_read_enabled": False}
                return invoke
            with self.subTest(failed_index=failed_index), ExitStack() as stack:
                for index, stage in enumerate(stages):
                    stack.enter_context(patch.object(stage, "process", handler(index)))
                with self.assertRaises(EmbeddingError):
                    build_import_graph(self.config).invoke({})
                self.assertEqual(visited, list(range(failed_index + 1)))


if __name__ == "__main__":
    unittest.main()
