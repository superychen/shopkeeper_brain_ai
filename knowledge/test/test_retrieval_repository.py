"""验证两条 ANN 都绑定商品/profile，并使用真实 SDK 请求对象。"""
import os
import unittest
import uuid
from unittest.mock import Mock
from pymilvus import MilvusClient
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.retrieval_config import RetrievalConfig
from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.service.chunk_search_repository import ChunkSearchRepository, FIELDS
from knowledge.service.chunk_repository import ChunkRepository, TEXT_FIELDS
from knowledge.service.item_name_embedding_service import EmbeddingVectors


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RetrievalConfig(query=QueryConfig(shared=ImportConfig(embedding_dim=3)))
        self.client = Mock()
        self.repo = ChunkSearchRepository(self.cfg, self.client)
        self.repo._ready = self.client
        self.vectors = EmbeddingVectors([1., 0., 0.], {1: 1.})
        self.client.query.return_value = []

    def test_bound_filters_and_low_score_retained(self):
        item = 'RS"\\12'
        entity = {k: "demo" for k in FIELDS} | dict(chunk_id="a"*64, item_name=item,
                    embedding_profile="profile", content="真实正文", chunk_index=1, metadata={})
        self.client.hybrid_search.return_value = [[{"entity": entity, "distance": .4}]]
        hits, warnings = self.repo.search(self.vectors, "profile", item, "vector")
        self.assertEqual(hits[0]["score"], .4)
        args = self.client.hybrid_search.call_args.kwargs
        self.assertEqual(args["limit"], 5)
        for req, metric in zip(args["reqs"], ("COSINE", "IP")):
            self.assertNotIn(item, req.expr)
            self.assertEqual(req.expr_params, {"item_names": [item], "profile": "profile"})
            self.assertEqual(req.param, {"metric_type": metric})
            self.assertEqual(req.limit, 50)

    def test_incompatible_profile_is_failure(self):
        self.client.hybrid_search.return_value = [[]]
        self.client.query.side_effect = [[{"chunk_id": "old"}], []]
        with self.assertRaisesRegex(QueryError, "模型空间"):
            self.repo.search(self.vectors, "new", "A", "vector")

    def test_out_of_scope_hit_rejected(self):
        self.client.hybrid_search.return_value = [[{"entity": {"item_name": "other"}, "distance": .8}]]
        with self.assertRaises(QueryError):
            self.repo.search(self.vectors, "p", "A", "vector")


@unittest.skipUnless(os.getenv("RUN_RETRIEVAL_MILVUS_INTEGRATION") == "1", "真实正文 Milvus 测试需显式启用")
class RetrievalMilvusTests(unittest.TestCase):
    def test_real_filter_and_response(self):
        collection = "it_retrieval_" + uuid.uuid4().hex
        shared = ImportConfig(embedding_dim=3, chunks_collection=collection)
        cfg = RetrievalConfig(query=QueryConfig(shared=shared))
        client = MilvusClient(uri=shared.milvus_url, token=shared.milvus_token or "", timeout=10)
        try:
            ChunkRepository(shared).ensure_collection(client)
            rows = []
            for index, item, profile in [(1, 'A"\\商品', "p"), (2, "其他商品", "p"), (3, 'A"\\商品', "old")]:
                rows.append({k: "demo" for k in TEXT_FIELDS} | dict(chunk_id=str(index)*64,
                    item_name=item, embedding_profile=profile, embedding_model=shared.bge_m3_model_name,
                    content="真实测试正文", chunk_index=index, updated_at=1, metadata={},
                    dense_vector=[1., 0., 0.], sparse_vector={1: 1.}))
            client.upsert(collection_name=collection, data=rows, timeout=10)
            hits, warnings = ChunkSearchRepository(cfg, client).search(
                EmbeddingVectors([1., 0., 0.], {1: 1.}), "p", 'A"\\商品', "vector")
            self.assertEqual([h["chunk_id"] for h in hits], ["1"*64])
            self.assertIn("incompatible_embedding_profile", warnings)
        finally:
            # 只删除本测试 UUID 集合，永远不操作业务集合。
            if client.has_collection(collection_name=collection, timeout=10):
                client.drop_collection(collection_name=collection, timeout=10)
            client.close()
