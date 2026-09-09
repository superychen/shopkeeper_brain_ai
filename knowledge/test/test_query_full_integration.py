"""真实提取、BGE-M3 与 Milvus 节点验收，仅写本次临时集合。"""

import os
import unittest
import uuid

from pymilvus import MilvusClient

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.main_graph import run_item_name_confirm
from knowledge.processor.query_processor.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.service.item_name_embedding_service import ItemNameEmbeddingService
from knowledge.service.item_name_repository import ItemNameRepository
from knowledge.service.item_name_search_repository import ItemNameSearchRepository


@unittest.skipUnless(os.getenv("RUN_QUERY_FULL_INTEGRATION") == "1", "完整查询验收会调用 DeepSeek 并加载 BGE-M3")
class QueryFullIntegrationTest(unittest.TestCase):
    def test_query_node_with_real_models_and_database(self):
        collection = "it_query_full_" + uuid.uuid4().hex
        shared = ImportConfig(item_name_collection=collection)
        cfg = QueryConfig(shared=shared)
        client = MilvusClient(uri=shared.milvus_url, token=shared.milvus_token or "", timeout=10)
        names = ["RS PRO RS-12 数字万用表", "华为擎云 L420x 笔记本电脑"]
        try:
            # 使用导入端的 document 编码，真正验证读写两端模型表示一致。
            vectors = [ItemNameEmbeddingService(shared).embed(name) for name in names]
            ItemNameRepository(shared).create_collection(client, collection)
            client.upsert(collection_name=collection, data=[
                dict(pk=f"p{i}", document_id=f"d{i}", item_name=name, file_title=name,
                     embedding_model=shared.bge_m3_model_name, updated_at=1,
                     dense_vector=vector.dense, sparse_vector=vector.sparse)
                for i, (name, vector) in enumerate(zip(names, vectors, strict=True))], timeout=10)
            node = ItemNameConfirmNode(cfg, repository=ItemNameSearchRepository(cfg, client))
            single = run_item_name_confirm({"original_query": "RS PRO RS-12 数字万用表怎么测电阻？"}, node=node)
            self.assertEqual(single["item_confirm_status"], "confirmed", single.get("error_code"))
            self.assertEqual(single["item_names"], [names[0]])
            self.assertGreaterEqual(single["confirmed_items"][0]["score"], .7)
            both = run_item_name_confirm({"original_query": "RS PRO RS-12 数字万用表怎么测电阻？华为擎云 L420x 笔记本电脑有什么使用要求？"}, node=node)
            self.assertEqual(both["item_confirm_status"], "confirmed", both.get("error_code"))
            self.assertEqual(set(both["item_names"]), set(names))
            missing = run_item_name_confirm({"original_query": "这个怎么用？"}, node=node)
            self.assertEqual(missing["item_confirm_status"], "needs_clarification")
            print("Real query validation:", single["item_confirm_status"], both["item_confirm_status"], missing["item_confirm_status"])
        finally:
            if client.has_collection(collection_name=collection, timeout=10):
                client.drop_collection(collection_name=collection, timeout=10)
            client.close()


if __name__ == "__main__":
    unittest.main()
