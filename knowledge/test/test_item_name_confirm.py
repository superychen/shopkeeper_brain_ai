"""验证业务边界和依赖调用契约，不访问外部服务。"""

import unittest
from unittest.mock import Mock, patch

from pymilvus import DataType

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.processor.query_processor.main_graph import run_item_name_confirm, route_after_item_confirm
from knowledge.processor.query_processor.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.schema.query_models import ItemHit, ItemSearchResult, QueryInput, QueryItemExtraction, QueryMention
from knowledge.service.item_name_aligner import ItemNameAligner, model_conflict
from knowledge.service.item_name_extractor import ItemNameExtractor
from knowledge.service.item_name_embedding_service import EmbeddingVectors
from knowledge.service.item_name_search_repository import ItemNameSearchRepository
from knowledge.service.query_embedding_service import QueryEmbeddingService
from knowledge.utils.client.ai_clients import AIClients


def config(**kwargs):
    return QueryConfig(shared=ImportConfig(embedding_dim=3, bge_m3_model_name="test-model",
                       item_name_collection="test_items", milvus_timeout_seconds=1), **kwargs)


def mention(name="RS-12", **kwargs):
    return QueryMention(name=name, evidence=name, source="current_query", **kwargs)


def hit(name="RS-12", score=0.8, pk="p1", document="d1"):
    return ItemHit(pk=pk, document_id=document, item_name=name, file_title="说明书",
                   embedding_model="test-model", score=score)


class AlignerTest(unittest.TestCase):
    def setUp(self):
        self.aligner = ItemNameAligner(config())

    def align(self, hits, name="RS-12", **kwargs):
        return self.aligner.align(mention(name), ItemSearchResult(hits=hits, **kwargs), "m1")

    def test_confirmation_and_candidate_boundaries(self):
        for score, status in ((0.7, "confirmed"), (0.699999, "needs_clarification"),
                              (0.45, "needs_clarification"), (0.449999, "not_found")):
            with self.subTest(score=score):
                self.assertEqual(self.align([hit(score=score)])["status"], status)

    def test_close_scores_on_opposite_sides_of_threshold(self):
        result = self.align([hit("华为笔记本甲", .701), hit("华为笔记本乙", .699, "p2")], "华为本")
        self.assertEqual(result["reason"], "close_candidates")

    def test_decimal_gap_boundary(self):
        for second, expected in ((.75, "confirmed"), (.750001, "needs_clarification"), (.74, "confirmed")):
            with self.subTest(second=second):
                result = self.align([hit("商品甲", .90), hit("商品乙", second, "p2")], "商品别名")
                self.assertEqual(result["status"], expected)

    def test_exact_match_wins_only_above_threshold(self):
        self.assertEqual(self.align([hit("RS-12", .7), hit("RS-13", .9, "p2")])["confirmed"]["item_name"], "RS-12")
        self.assertNotEqual(self.align([hit("RS-12", .69), hit("RS-13", .9, "p2")])["status"], "confirmed")

    def test_model_suffix_conflict(self):
        self.assertTrue(model_conflict("L420", "华为 L420x 笔记本"))
        self.assertTrue(model_conflict("DT-9205A", "DT-9205B"))
        self.assertFalse(model_conflict("rs-12", "RS PRO RS-12 数字万用表"))
        self.assertEqual(self.align([hit("L420x", .99)], "L420")["reason"], "model_conflict")

    def test_duplicate_documents_are_one_candidate(self):
        result = self.align([hit("RS-12 数字万用表", .80), hit("RS-12 数字万用表", .80, "p2", "d2")])
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(len(result["candidates"]), 1)
        self.assertEqual(result["confirmed"]["matched_document_ids"], ["d1", "d2"])

    def test_truncated_recall_never_auto_confirms(self):
        self.assertEqual(self.align([hit(score=.99)], truncated=True)["reason"], "recall_truncated")

    def test_two_explicit_products_are_not_globally_filtered(self):
        rows = [self.align([hit("RS-12", .95)]),
                self.aligner.align(mention("UT61E"), ItemSearchResult(hits=[hit("UT61E", .78, "p2")]), "m2")]
        result = self.aligner.aggregate(rows, "RS-12 和 UT61E 有什么区别，不要省略注意事项")
        self.assertEqual(result["item_names"], ["RS-12", "UT61E"])
        self.assertIn("不要省略注意事项", result["rewritten_query"])

    def test_partial_success_blocks_downstream(self):
        rows = [self.align([hit()]), self.aligner.align(mention("未知商品"), ItemSearchResult(hits=[]), "m2")]
        result = self.aligner.aggregate(rows, "比较两款商品")
        self.assertEqual(result["item_confirm_status"], "needs_clarification")
        self.assertEqual(result["item_names"], [])
        self.assertEqual(len(result["confirmed_items"]), 1)
        self.assertEqual(route_after_item_confirm(result), "end")

    def test_candidates_are_allocated_across_mentions(self):
        rows = [self.aligner.align(mention(f"别名{i}"), ItemSearchResult(hits=[
            hit(f"产品{j}", .6, f"p{j}") for j in range(5)]), f"m{i}") for i in range(5)]
        result = self.aligner.aggregate(rows, "五个名称")
        self.assertEqual(sum(len(g["candidates"]) for g in result["options"]), 5)
        self.assertTrue(all(len(g["candidates"]) == 1 and g["has_more"] for g in result["options"]))


class NodeTest(unittest.TestCase):
    def node(self, names=("RS-12",), hits=None):
        extractor = Mock()
        extractor.extract.return_value = QueryItemExtraction(mentions=[mention(n) for n in names], rewritten_query="草稿")
        embedding = Mock()
        embedding.embed.return_value = [EmbeddingVectors([1., 0., 0.], {1: 1.}) for _ in names]
        repository = Mock()
        repository.search.return_value = ItemSearchResult(hits=[hit()] if hits is None else hits)
        return ItemNameConfirmNode(config(), extractor=extractor, embedding=embedding, repository=repository)

    def test_graph_clears_previous_answer(self):
        result = run_item_name_confirm({"original_query": "RS-12 怎么使用", "answer": "旧回答",
                                      "options": [{"old": True}], "error_code": "old"}, node=self.node())
        self.assertEqual(result["item_names"], ["RS-12"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["options"], [])
        self.assertEqual(result["error_code"], "")
        self.assertEqual(route_after_item_confirm(result), "retrieve")

    def test_no_mention_skips_embedding_and_database(self):
        node = self.node(names=())
        result = node.process({"original_query": "它怎么用", "item_names": ["old"]})
        self.assertEqual(result["item_confirm_status"], "needs_clarification")
        self.assertEqual(result["item_names"], [])
        node.embedding.embed.assert_not_called()
        node.repository.search.assert_not_called()

    def test_generic_name_is_not_guessed(self):
        node = self.node(names=("万用表",))
        result = node.process({"original_query": "万用表怎么用"})
        self.assertEqual(result["item_confirm_status"], "needs_clarification")
        node.repository.search.assert_not_called()

    def test_too_many_products_requests_split_without_partial_search(self):
        node = self.node(names=())
        node.extractor.extract.return_value = QueryItemExtraction(mentions=[], rewritten_query="多个商品",
                                                                 too_many_products=True)
        result = node.process({"original_query": "请比较六个型号"})
        self.assertEqual(result["warnings"], ["too_many_products"])
        self.assertEqual(result["item_confirm_status"], "needs_clarification")
        node.embedding.embed.assert_not_called()
        node.repository.search.assert_not_called()

    def test_external_failure_is_not_no_match(self):
        node = self.node()
        node.repository.search.side_effect = QueryError("milvus_unavailable", "检索失败")
        result = node.process({"original_query": "RS-12", "item_names": ["old"], "rewritten_query": "old"})
        self.assertEqual(result["item_confirm_status"], "error")
        self.assertEqual(result["error_code"], "milvus_unavailable")
        self.assertEqual(result["item_names"], [])
        self.assertEqual(result["rewritten_query"], "")

    def test_invalid_input_and_history(self):
        for state in ({}, {"original_query": "  "}, {"original_query": 1}, {"original_query": "x" * 2001},
                      {"original_query": "x", "request_id": "unsafe\nlog"},
                      {"original_query": "x", "history": [{"role": "system", "content": "x"}]}):
            with self.subTest(state=str(state)[:60]):
                node = self.node()
                result = node.process(state)
                self.assertEqual(result["error_code"], "invalid_input")
                node.extractor.extract.assert_not_called()

    def test_full_failure_discards_partial_success(self):
        node = self.node(names=("RS-12", "UT61E"))
        node.repository.search.side_effect = [ItemSearchResult(hits=[hit()]), QueryError("milvus_unavailable", "失败")]
        result = node.process({"original_query": "RS-12 与 UT61E"})
        self.assertEqual(result["item_confirm_status"], "error")
        self.assertEqual(result["confirmed_items"], [])


class ExtractorTest(unittest.TestCase):
    def test_structured_extraction_validates_evidence_and_repairs_once(self):
        llm = Mock()
        llm.with_structured_output.return_value.invoke.side_effect = [
            {"mentions": [{"name": "L420x", "evidence": "L420", "source": "current_query"}], "rewritten_query": "x"},
            {"mentions": [{"name": "L420", "evidence": "L420", "source": "current_query"}], "rewritten_query": "x"}]
        result = ItemNameExtractor(config(), llm).extract(QueryInput(original_query="L420 怎么用"))
        self.assertEqual(result.mentions[0].name, "L420")
        self.assertEqual(llm.with_structured_output.return_value.invoke.call_count, 2)

    def test_invalid_output_is_not_empty_extraction(self):
        llm = Mock()
        llm.with_structured_output.return_value.invoke.return_value = {"mentions": None}
        with self.assertRaises(QueryError) as caught:
            ItemNameExtractor(config(), llm).extract(QueryInput(original_query="RS-12"))
        self.assertEqual(caught.exception.code, "llm_invalid_output")

    def test_history_evidence_cannot_be_invented(self):
        llm = Mock()
        llm.with_structured_output.return_value.invoke.return_value = QueryItemExtraction(
            mentions=[QueryMention(name="RS-12", evidence="RS-12", source="history")], rewritten_query="x")
        with self.assertRaises(QueryError):
            ItemNameExtractor(config(), llm).extract(QueryInput(original_query="它怎么用"))

    def test_embedding_uses_query_encoder(self):
        model = Mock()
        model.encode_queries.return_value = {"dense": [[1., 0., 0.]], "sparse": [{1: 1.}]}
        with patch.object(AIClients, "get_bge_m3", return_value=model):
            vectors = QueryEmbeddingService(config()).embed(["RS-12"])
        self.assertEqual(vectors[0].sparse, {1: 1.})
        model.encode_queries.assert_called_once_with(["RS-12"])
        model.encode_documents.assert_not_called()

    def test_cached_embedding_model_must_match_configuration(self):
        model = Mock(model_name="different-model")
        with patch.object(AIClients, "get_bge_m3", return_value=model), self.assertRaises(QueryError) as caught:
            QueryEmbeddingService(config()).embed(["RS-12"])
        self.assertEqual(caught.exception.code, "embedding_failed")
        model.encode_queries.assert_not_called()


def fake_client():
    client = Mock()
    fields = []
    for name, size in (("pk", 64), ("document_id", 128), ("file_title", 1024), ("item_name", 1024), ("embedding_model", 128)):
        fields.append({"name": name, "type": DataType.VARCHAR, "params": {"max_length": size}, "is_primary": name == "pk"})
    fields += [{"name": "updated_at", "type": DataType.INT64},
               {"name": "dense_vector", "type": DataType.FLOAT_VECTOR, "params": {"dim": 3}},
               {"name": "sparse_vector", "type": DataType.SPARSE_FLOAT_VECTOR}]
    client.has_collection.return_value = True
    client.describe_collection.return_value = {"auto_id": False, "fields": fields}
    client.list_indexes.return_value = ["dense_vector_index", "sparse_vector_index"]
    client.describe_index.side_effect = lambda **kw: {"metric_type": "COSINE" if kw["index_name"].startswith("dense") else "IP"}
    client.query.return_value = []
    client.hybrid_search.return_value = [[raw_hit()]]
    return client


def raw_hit(name="RS-12", score=.8, pk="p1"):
    entity = hit(name, .8, pk).model_dump()
    entity.pop("score")
    return {"id": pk, "distance": score, "entity": entity}


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self.client = fake_client()
        self.repo = ItemNameSearchRepository(config(), self.client)
        self.vectors = EmbeddingVectors([1., 0., 0.], {1: 1.})

    def test_hybrid_contract_and_read_only_operations(self):
        result = self.repo.search(self.vectors)
        self.assertEqual(result.hits[0].score, .8)
        call = self.client.hybrid_search.call_args.kwargs
        self.assertEqual(call["ranker"].dict(), {"strategy": "weighted", "params": {"weights": [.5, .5], "norm_score": True}})
        self.assertEqual([r.anns_field for r in call["reqs"]], ["dense_vector", "sparse_vector"])
        self.assertEqual([r.param["metric_type"] for r in call["reqs"]], ["COSINE", "IP"])
        self.assertEqual(call["round_decimal"], -1)
        self.assertEqual(call["consistency_level"], "Strong")
        for method in ("create_collection", "upsert", "delete", "drop_collection"):
            getattr(self.client, method).assert_not_called()
        self.repo.search(self.vectors)
        self.assertEqual(self.client.load_collection.call_count, 1)

    def test_missing_collection_differs_from_empty_collection(self):
        self.client.has_collection.return_value = False
        with self.assertRaises(QueryError) as caught:
            self.repo.search(self.vectors)
        self.assertEqual(caught.exception.code, "knowledge_not_ready")
        self.client.has_collection.return_value = True
        self.client.hybrid_search.return_value = [[]]
        self.assertEqual(self.repo.search(self.vectors).hits, [])

    def test_incompatible_model_without_compatible_records(self):
        self.client.query.side_effect = [[{"pk": "foreign"}], []]
        self.client.hybrid_search.return_value = [[]]
        with self.assertRaises(QueryError) as caught:
            self.repo.search(self.vectors)
        self.assertEqual(caught.exception.code, "knowledge_not_ready")

    def test_recall_expands_and_reports_saturation(self):
        self.repo = ItemNameSearchRepository(config(item_name_top_k=5, item_name_max_top_k=10), self.client)
        self.client.hybrid_search.side_effect = lambda **kw: [[raw_hit(pk=f"p{i}") for i in range(kw["limit"])]]
        result = self.repo.search(self.vectors)
        self.assertTrue(result.truncated)
        self.assertEqual([c.kwargs["limit"] for c in self.client.hybrid_search.call_args_list], [5, 10])

    def test_bad_scores_and_shapes_are_errors(self):
        for raw in ([], [[raw_hit(score=float("nan"))]], [[raw_hit(score=1.01)]], [[{"entity": {}}]]):
            with self.subTest(raw=raw):
                self.client.hybrid_search.return_value = raw
                with self.assertRaises(QueryError) as caught:
                    self.repo.search(self.vectors)
                self.assertEqual(caught.exception.code, "invalid_search_result")

    def test_bad_index_is_configuration_error(self):
        self.client.describe_index.side_effect = lambda **kw: {"metric_type": "L2"}
        with self.assertRaises(QueryError) as caught:
            self.repo.search(self.vectors)
        self.assertEqual(caught.exception.code, "schema_incompatible")

    def test_network_failure_has_distinct_code(self):
        self.client.hybrid_search.side_effect = TimeoutError("do not expose this message")
        with self.assertRaises(QueryError) as caught:
            self.repo.search(self.vectors)
        self.assertEqual(caught.exception.code, "milvus_unavailable")
        self.assertNotIn("do not expose", str(caught.exception))


class ConfigTest(unittest.TestCase):
    def test_invalid_parameters(self):
        for values in ({"item_name_mid_confidence": .7}, {"item_name_score_gap": float("nan")},
                       {"item_name_dense_weight": .7}, {"item_name_top_k": 0}, {"item_name_max_extracted": 6}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                config(**values)

    def test_confirmation_threshold_cannot_be_overridden(self):
        with self.assertRaises(TypeError):
            config(item_name_high_confidence=.6)


if __name__ == "__main__":
    unittest.main()
