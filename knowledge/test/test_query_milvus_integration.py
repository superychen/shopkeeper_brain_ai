"""隔离集合验证真实服务端 WeightedRanker 分数，默认不连接外部服务。"""

import math
import os
import unittest
import uuid

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.service.item_name_embedding_service import EmbeddingVectors
from knowledge.service.item_name_repository import ItemNameRepository
from knowledge.service.item_name_search_repository import ItemNameSearchRepository


@unittest.skipUnless(os.getenv("RUN_QUERY_MILVUS_INTEGRATION") == "1", "查询 Milvus 验收需显式启用")
class QueryMilvusIntegrationTest(unittest.TestCase):
    def test_server_hybrid_score_and_repository_contract(self):
        collection = "it_query_items_" + uuid.uuid4().hex
        cfg = ImportConfig(embedding_dim=3, item_name_collection=collection)
        client = MilvusClient(uri=cfg.milvus_url, token=cfg.milvus_token or "", timeout=10)
        try:
            print("Query integration server:", client.get_server_version(timeout=10))
            ItemNameRepository(cfg).create_collection(client, collection)
            client.upsert(collection_name=collection, data=[
                dict(pk="p1", document_id="d1", item_name="RS-12", file_title="manual",
                     embedding_model=cfg.bge_m3_model_name, updated_at=1,
                     dense_vector=[1., 0., 0.], sparse_vector={1: 1.}),
                dict(pk="p2", document_id="d2", item_name="UT61E", file_title="manual",
                     embedding_model=cfg.bge_m3_model_name, updated_at=1,
                     dense_vector=[0., 1., 0.], sparse_vector={2: 1.})], timeout=10)
            vectors = EmbeddingVectors([1., 0., 0.], {1: 1.})
            repository = ItemNameSearchRepository(QueryConfig(shared=cfg), client)
            result = repository.search(vectors)
            self.assertEqual(result.hits[0].pk, "p1")
            self.assertAlmostEqual(result.hits[0].score, .875, places=5)
            # 从实际单路候选重建期望分数，尤其验证未召回通道贡献为零。
            args = dict(collection_name=collection, limit=10, timeout=10, consistency_level="Strong")
            dense = client.search(**args, output_fields=["pk"], data=[vectors.dense], anns_field="dense_vector",
                                  search_params={"metric_type": "COSINE"})[0]
            sparse = client.search(**args, output_fields=["pk"], data=[vectors.sparse], anns_field="sparse_vector",
                                   search_params={"metric_type": "IP"})[0]
            expected = {}
            for row in dense:
                expected[row["entity"]["pk"]] = .5 * (1 + row["distance"]) / 2
            for row in sparse:
                pk = row["entity"]["pk"]
                expected[pk] = expected.get(pk, 0) + .5 * (.5 + math.atan(row["distance"]) / math.pi)
            for row in result.hits:
                self.assertAlmostEqual(row.score, expected[row.pk], places=5)
            unnormalized = client.hybrid_search(**args,
                reqs=[AnnSearchRequest(data=[vectors.dense], anns_field="dense_vector", param={"metric_type": "COSINE"}, limit=10),
                      AnnSearchRequest(data=[vectors.sparse], anns_field="sparse_vector", param={"metric_type": "IP"}, limit=10)],
                ranker=WeightedRanker(.5, .5, norm_score=False))
            self.assertAlmostEqual(unnormalized[0][0]["distance"], 1., places=5)
            self.assertFalse(result.truncated)
        finally:
            # 仅清理本测试随机命名的集合，永远不删除业务集合。
            if client.has_collection(collection_name=collection, timeout=10):
                client.drop_collection(collection_name=collection, timeout=10)
            client.close()


if __name__ == "__main__":
    unittest.main()
