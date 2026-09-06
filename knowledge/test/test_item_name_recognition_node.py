"""ItemNameRecognitionNode 的纯本地单元测试。"""

import hashlib
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda
from pymilvus import DataType

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    EmbeddingError,
    LLMError,
    MilvusError,
)
from knowledge.processor.import_processor.nodes.item_name_recognition_node import (
    ItemNameRecognitionNode,
)
from knowledge.schema.item_name import ItemNameExtraction
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients


class _FakeRecognitionChain:
    """替代真实 LangChain，只记录模板变量并返回已校验模型。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.inputs: list[dict[str, str]] = []

    def invoke(self, values: dict[str, str]) -> ItemNameExtraction:
        self.inputs.append(values)
        return ItemNameExtraction.model_validate(self.payload)


class _FakeStructuredLlm:
    """验证节点确实通过 LangChain 的结构化输出能力组链。"""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.schema = None
        self.method = None

    def with_structured_output(self, schema, *, method: str):
        self.schema = schema
        self.method = method
        return RunnableLambda(
            lambda _prompt: ItemNameExtraction.model_validate(self.payload)
        )


class _FakeBgeM3:
    def encode_documents(self, documents: list[str]) -> dict:
        if len(documents) != 1:
            raise AssertionError("商品名称节点每次只应编码一个名称")
        return {
            "dense": [[0.1, 0.2, 0.3]],
            "sparse": [{12: 0.7, 98: 0.2}],
        }


class _FakeSchema:
    def __init__(self) -> None:
        self.fields: list[dict] = []

    def add_field(self, **kwargs) -> None:
        field = dict(kwargs)
        field["type"] = int(field.pop("datatype"))
        field.setdefault("is_primary", False)
        params = {}
        if "dim" in field:
            params["dim"] = str(field.pop("dim"))
        if "max_length" in field:
            params["max_length"] = str(field.pop("max_length"))
        field["params"] = params
        field["name"] = field.pop("field_name")
        self.fields.append(field)


class _FakeIndexParams:
    def __init__(self) -> None:
        self.indexes: dict[str, dict] = {}

    def add_index(self, **kwargs) -> None:
        self.indexes[kwargs["index_name"]] = dict(kwargs)


class _FakeMilvus:
    def __init__(self) -> None:
        self.exists = False
        self.schema: _FakeSchema | None = None
        self.indexes: dict[str, dict] = {}
        self.upserts: list[dict] = []

    def has_collection(self, *, collection_name: str) -> bool:
        return self.exists

    def create_schema(self, **_kwargs) -> _FakeSchema:
        return _FakeSchema()

    def prepare_index_params(self) -> _FakeIndexParams:
        return _FakeIndexParams()

    def create_collection(
        self,
        *,
        collection_name: str,
        schema: _FakeSchema,
        index_params: _FakeIndexParams,
    ) -> None:
        self.exists = True
        self.schema = schema
        self.indexes = index_params.indexes

    def describe_collection(self, *, collection_name: str) -> dict:
        if self.schema is None:
            raise AssertionError("测试集合尚未创建")
        return {"auto_id": False, "fields": self.schema.fields}

    def list_indexes(self, *, collection_name: str) -> list[str]:
        return list(self.indexes)

    def describe_index(self, *, collection_name: str, index_name: str) -> dict:
        return {"metric_type": self.indexes[index_name]["metric_type"]}

    def upsert(self, *, collection_name: str, data: list[dict]) -> dict:
        self.upserts.extend(data)
        return {"upsert_count": len(data)}


class ItemNameRecognitionNodeTest(unittest.TestCase):
    """覆盖识别契约、混合向量、幂等写入和状态发布。"""

    def setUp(self) -> None:
        self.config = ImportConfig(
            item_name_chunk_k=3,
            item_name_chunk_size=500,
            deepseek_api_base="https://api.deepseek.test",
            deepseek_api_key="test-key",
            deepseek_llm_model="deepseek-test",
            deepseek_timeout_seconds=1,
            deepseek_max_retries=0,
            bge_m3_model_name="BAAI/bge-m3",
            bge_m3_device="cpu",
            embedding_dim=3,
            milvus_url="http://milvus.test:19530",
            item_name_collection="kb_item_names_v1",
        )
        self.node = ItemNameRecognitionNode(config=self.config)

    @staticmethod
    def _recognized_payload() -> dict:
        return {
            "status": "recognized",
            "item_name": "RS PRO RS-12 数字万用表",
            "brand": "RS PRO",
            "model": "RS-12",
            "product_type": "数字万用表",
            "confidence": 0.98,
            "evidence": ["RS PRO", "RS-12", "数字万用表"],
        }

    @staticmethod
    def _state() -> dict:
        return {
            "task_id": "task-1",
            "document_id": "document-1",
            "file_title": "万用表的使用",
            "chunks": [
                {
                    "chunk_index": 0,
                    "title": "万用表的使用",
                    "content": "RS PRO\n使用说明书\nRS-12\n数字万用表",
                },
                {
                    "chunk_index": 1,
                    "title": "安全手册",
                    "content": "使用前请阅读安全说明。",
                },
            ],
        }

    def test_context_keeps_complete_chunks_and_truncates_oversized_first(self) -> None:
        inputs = self.node._validate_inputs(self._state())
        context = self.node._build_recognition_context(inputs)

        self.assertIn("RS-12", context)
        self.assertIn("安全手册", context)

        small_config = ImportConfig(**{**self.config.__dict__, "item_name_chunk_size": 100})
        small_node = ItemNameRecognitionNode(config=small_config)
        state = self._state()
        state["chunks"][0]["content"] = "超长正文" * 100
        truncated = small_node._build_recognition_context(
            small_node._validate_inputs(state)
        )

        self.assertLessEqual(len(truncated), 100)
        self.assertIn("[上下文已截断]", truncated)

    def test_langchain_uses_json_mode_and_validates_source_evidence(self) -> None:
        fake_llm = _FakeStructuredLlm(self._recognized_payload())
        context = self.node._build_recognition_context(
            self.node._validate_inputs(self._state())
        )

        with patch.object(AIClients, "get_llm", return_value=fake_llm):
            result = self.node._recognize_with_deepseek("万用表的使用", context)

        self.assertEqual(result.item_name, "RS PRO RS-12 数字万用表")
        self.assertIs(fake_llm.schema, ItemNameExtraction)
        self.assertEqual(fake_llm.method, "json_mode")

    def test_llm_client_uses_official_langchain_deepseek_integration(self) -> None:
        with patch("knowledge.utils.client.ai_clients.ChatDeepSeek") as chat_deepseek:
            client = AIClients._create_llm(self.config)

        self.assertIs(client, chat_deepseek.return_value)
        options = chat_deepseek.call_args.kwargs
        self.assertEqual(options["model"], "deepseek-test")
        self.assertEqual(options["base_url"], "https://api.deepseek.test")
        self.assertEqual(options["max_retries"], 0)
        self.assertEqual(options["extra_body"], {"thinking": {"type": "disabled"}})

    def test_hallucinated_evidence_is_rejected(self) -> None:
        payload = self._recognized_payload()
        payload["evidence"] = ["原文中不存在的型号"]

        with self.assertRaises(LLMError):
            self.node._validate_extraction(
                ItemNameExtraction.model_validate(payload),
                source_text="RS PRO RS-12 数字万用表",
            )

    def test_not_found_skips_embedding_and_milvus(self) -> None:
        fake_chain = _FakeRecognitionChain(
            {
                "status": "not_found",
                "item_name": "",
                "brand": "",
                "model": "",
                "product_type": "",
                "confidence": 0.0,
                "evidence": [],
            }
        )
        state = self._state()

        with (
            patch.object(
                self.node,
                "_get_recognition_chain",
                return_value=fake_chain,
            ),
            patch.object(AIClients, "get_bge_m3") as get_bge_m3,
            patch.object(StorageClients, "get_milvus") as get_milvus,
        ):
            result = self.node.process(state)

        self.assertEqual(result["item_name_status"], "not_found")
        self.assertEqual(result["item_name"], "")
        self.assertNotIn("item_name", result["chunks"][0])
        get_bge_m3.assert_not_called()
        get_milvus.assert_not_called()

    def test_success_creates_collection_upserts_and_publishes_new_chunks(self) -> None:
        fake_chain = _FakeRecognitionChain(self._recognized_payload())
        fake_milvus = _FakeMilvus()
        state = self._state()
        original_chunks = state["chunks"]
        original_first_chunk = original_chunks[0]

        with (
            patch.object(
                self.node,
                "_get_recognition_chain",
                return_value=fake_chain,
            ),
            patch.object(AIClients, "get_bge_m3", return_value=_FakeBgeM3()),
            patch.object(StorageClients, "get_milvus", return_value=fake_milvus),
        ):
            result = self.node.process(state)

        expected_pk = hashlib.sha256(b"item-name:v1\0document-1").hexdigest()
        self.assertEqual(result["item_name_status"], "recognized")
        self.assertEqual(result["item_name_milvus_pk"], expected_pk)
        self.assertEqual(result["chunks"][0]["item_name"], "RS PRO RS-12 数字万用表")
        self.assertIsNot(result["chunks"], original_chunks)
        self.assertNotIn("item_name", original_first_chunk)
        self.assertEqual(fake_milvus.upserts[0]["pk"], expected_pk)
        self.assertEqual(fake_milvus.upserts[0]["dense_vector"], [0.1, 0.2, 0.3])
        self.assertEqual(fake_milvus.upserts[0]["sparse_vector"], {12: 0.7, 98: 0.2})
        self.assertEqual(
            fake_milvus.indexes["dense_vector_index"]["metric_type"],
            "COSINE",
        )
        self.assertEqual(
            fake_milvus.indexes["sparse_vector_index"]["metric_type"],
            "IP",
        )

    def test_repeated_document_uses_same_primary_key(self) -> None:
        fake_chain = _FakeRecognitionChain(self._recognized_payload())
        fake_milvus = _FakeMilvus()
        with (
            patch.object(
                self.node,
                "_get_recognition_chain",
                return_value=fake_chain,
            ),
            patch.object(AIClients, "get_bge_m3", return_value=_FakeBgeM3()),
            patch.object(StorageClients, "get_milvus", return_value=fake_milvus),
        ):
            self.node.process(self._state())
            self.node.process(self._state())

        self.assertEqual(len(fake_milvus.upserts), 2)
        self.assertEqual(fake_milvus.upserts[0]["pk"], fake_milvus.upserts[1]["pk"])

    def test_milvus_failure_does_not_publish_success_state(self) -> None:
        fake_chain = _FakeRecognitionChain(self._recognized_payload())
        fake_milvus = _FakeMilvus()
        state = self._state()
        original_chunks = state["chunks"]

        with (
            patch.object(
                self.node,
                "_get_recognition_chain",
                return_value=fake_chain,
            ),
            patch.object(AIClients, "get_bge_m3", return_value=_FakeBgeM3()),
            patch.object(StorageClients, "get_milvus", return_value=fake_milvus),
            patch.object(fake_milvus, "upsert", side_effect=RuntimeError("offline")),
            self.assertRaises(MilvusError),
        ):
            self.node.process(state)

        self.assertNotIn("item_name_status", state)
        self.assertIs(state["chunks"], original_chunks)
        self.assertNotIn("item_name", state["chunks"][0])

    def test_vector_dimension_and_sparse_values_are_strictly_checked(self) -> None:
        with self.assertRaises(EmbeddingError):
            self.node._embedding_service.validate_vectors([0.1, 0.2], {1: 0.5})
        with self.assertRaises(EmbeddingError):
            self.node._embedding_service.validate_vectors(
                [0.1, 0.2, 0.3], {1: 0.0}
            )

    def test_existing_incompatible_collection_is_rejected(self) -> None:
        fake_milvus = _FakeMilvus()
        self.node._repository.create_collection(fake_milvus, "kb_item_names_v1")
        if fake_milvus.schema is None:
            self.fail("测试集合 Schema 未创建")
        dense_field = next(
            field for field in fake_milvus.schema.fields if field["name"] == "dense_vector"
        )
        dense_field["params"]["dim"] = "999"

        with self.assertRaises(ConfigurationError):
            self.node._repository.ensure_collection(fake_milvus, "kb_item_names_v1")


if __name__ == "__main__":
    unittest.main()
