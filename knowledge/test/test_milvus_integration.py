"""显式开启的真实 Milvus 验收，仅使用本次创建的隔离集合。"""

import hashlib
import os
import unittest
import uuid
from unittest.mock import patch

from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.nodes.milvus_import_node import MilvusImportNode
from knowledge.service.chunk_embedding_service import embedding_text
from knowledge.test.test_chunk_import import make_state
from knowledge.utils.client.storage_clients import StorageClients


@unittest.skipUnless(os.getenv("RUN_MILVUS_INTEGRATION") == "1", "真实 Milvus 验收需显式启用")
class MilvusIntegrationTest(unittest.TestCase):
    def test_real_schema_upsert_reimport_cleanup_and_hybrid_search(self):
        config = ImportConfig(embedding_dim=3, milvus_batch_size=2,
                              chunks_collection="it_chunks_" + uuid.uuid4().hex)
        client = MilvusClient(uri=config.milvus_url, token=config.milvus_token or "", timeout=10)
        print("Milvus integration server:", client.get_server_version(timeout=10))
        def state(count):
            result = make_state(count)
            for chunk in result["chunks"]:
                chunk.update(dense_vector=[0.1, 0.2, 0.3], sparse_vector={1: 0.7, 2: 0.3},
                             embedding_text_hash=hashlib.sha256(embedding_text(chunk["content"], "").encode()).hexdigest())
            return {**result, "embedded_chunk_count": count, "embedding_model": config.bge_m3_model_name,
                    "embedding_profile": "integration-synthetic-v1", "split_profile": "test-split"}
        try:
            with patch.object(StorageClients, "get_milvus", return_value=client):
                node = MilvusImportNode(config)
                first = node.process(state(5))
                repeated = node.process(state(5))
                self.assertEqual(first["milvus_chunk_ids"], repeated["milvus_chunk_ids"])
                final = node.process(state(3))
                self.assertEqual(final["written_chunk_count"], 3)
            hits = client.hybrid_search(collection_name=config.chunks_collection,
                                        reqs=[AnnSearchRequest(data=[[0.1, 0.2, 0.3]], anns_field="dense_vector",
                                                               param={"metric_type": "COSINE"}, limit=3),
                                              AnnSearchRequest(data=[{1: 0.7, 2: 0.3}], anns_field="sparse_vector",
                                                               param={"metric_type": "IP"}, limit=3)],
                                        ranker=RRFRanker(), limit=3, output_fields=["content", "document_id"],
                                        consistency_level="Strong", timeout=20)
            self.assertEqual(len(hits[0]), 3)
            self.assertTrue(all(hit["entity"]["document_id"] == "doc-1" for hit in hits[0]))
        finally:
            # 集合名由本测试生成，禁止以业务集合配置作为清理目标。
            if client.has_collection(collection_name=config.chunks_collection, timeout=10):
                client.drop_collection(collection_name=config.chunks_collection, timeout=10)
            client.close()


if __name__ == "__main__":
    unittest.main()
