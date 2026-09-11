"""三路契约测试：不访问模型、Milvus 或计费联网服务。"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
import unittest
from knowledge.processor.query_processor.retrieval_config import RetrievalConfig
from knowledge.processor.query_processor.retrieval_graph import arun_query_retrieval, OUTPUTS, METAS, join
from knowledge.processor.query_processor.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_processor.nodes.hyde_search_node import HyDESearchNode
from knowledge.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode
from knowledge.service.bailian_search_service import parse_pages
from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.service.bailian_search_service import BailianSearchService
from knowledge.processor.query_processor.retrieval_base import failure_code


def confirmed(state):
    return {"item_confirm_status": "confirmed", "item_names": ["RS-12"], "rewritten_query": "绑定说明"}


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_barrier_and_clear_stale(self):
        started = set()
        barrier = asyncio.Event()
        def branch(i):
            async def run(state):
                started.add(i)
                if len(started) == 3:
                    barrier.set()
                await asyncio.wait_for(barrier.wait(), 1)
                return {OUTPUTS[i]: [{"route": i}], METAS[i]: {"status": "success", "items": [], "warnings": []}}
            return run
        result = await arun_query_retrieval({"original_query": "怎么用", "embedding_chunks": ["stale"]},
            confirm_node=confirmed, nodes=[branch(i) for i in range(3)])
        self.assertEqual(result["retrieval_status"], "success")
        self.assertEqual(result["embedding_chunks"], [{"route": 0}])

    async def test_unconfirmed_never_calls_branches(self):
        async def forbidden(state):
            self.fail("未确认不能检索")
        result = await arun_query_retrieval({"original_query": "怎么用", "item_confirm_status": "confirmed", "item_names": ["fake"]},
            confirm_node=lambda s: {"item_confirm_status": "needs_clarification"}, nodes=[forbidden]*3)
        self.assertEqual(result["retrieval_status"], "skipped")
        self.assertEqual(result["embedding_chunks"], [])

    async def test_hyde_failure_does_not_retrieve(self):
        class Generator:
            async def generate(self, *args):
                raise QueryError("hyde_generation_failed", "failed")
        node = HyDESearchNode(RetrievalConfig(), None, None, Generator())
        result = await node({"retrieval_input": {"items": ["A"], "query": "问题"}})
        self.assertEqual(result["hyde_search_meta"]["status"], "failed")
        self.assertEqual(result["hyde_embedding_chunks"], [])

    async def test_timeout_preserves_completed_item(self):
        class Node(VectorSearchNode):
            async def retrieve(self, item, query):
                if item == "slow":
                    await asyncio.sleep(10)
                return [{"item_name": item}], []
        result = await Node(RetrievalConfig(vector_timeout=.05), None, None)(
            {"retrieval_input": {"items": ["fast", "slow"], "query": "q"}})
        self.assertEqual(result["vector_search_meta"]["status"], "partial")
        self.assertEqual(result["embedding_chunks"], [{"item_name": "fast"}])

    async def test_external_cancel_propagates(self):
        class Node(VectorSearchNode):
            async def retrieve(self, *args):
                raise asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await Node(RetrievalConfig(), None, None)({"retrieval_input": {"items": ["A"], "query": "q"}})

    async def test_disabled_web_no_session(self):
        result = await WebMcpSearchNode(RetrievalConfig(web_enabled=False), None)(
            {"retrieval_input": {"items": ["A"], "query": "q"}})
        self.assertEqual(result["web_search_meta"]["status"], "skipped")

    def test_status_matrix(self):
        for statuses, expected in [(["empty"]*3, "empty"), (["failed"]*3, "failed"),
                                   (["empty", "failed", "skipped"], "partial")]:
            state = {k: [] for k in OUTPUTS} | {k: {"status": s} for k, s in zip(METAS, statuses)}
            self.assertEqual(join(state)["retrieval_status"], expected)

    def test_web_structured_and_duplicates(self):
        page = {"title": "说明", "url": "https://example.com/a#top", "snippet": "正文"}
        hits, warnings = parse_pages(SimpleNamespace(isError=False, structuredContent={"pages": [page, page]}), "A", 3)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["url"], "https://example.com/a")
        self.assertIsNone(hits[0]["published_at"])

    def test_web_invalid_not_empty(self):
        for result in [SimpleNamespace(isError=True), SimpleNamespace(isError=False, structuredContent={"unexpected": []}),
                       SimpleNamespace(isError=False, structuredContent={"pages": [{"title": "x"}]})]:
            with self.assertRaises(QueryError):
                parse_pages(result, "A", 3)

    async def test_tool_schema_and_no_secret_external_call(self):
        from unittest.mock import AsyncMock
        session = SimpleNamespace(call_tool=AsyncMock(return_value=SimpleNamespace(
            isError=False, structuredContent={"pages": []})))
        service = BailianSearchService(RetrievalConfig(tool_name="search"))
        schema = {"type": "object", "properties": {"query": {"type": "string"},
                  "count": {"type": "integer", "maximum": 3}}, "required": ["query"]}
        hits, _ = await service.search((session, schema), "A", "怎么用")
        self.assertEqual(hits, [])
        self.assertEqual(session.call_tool.call_args.args, ("search", {"query": "A\n怎么用", "count": 3}))
        session.call_tool.reset_mock()
        with self.assertRaises(QueryError):
            await service.search((session, schema), "A", "password=private")
        session.call_tool.assert_not_called()
        schema["required"].append("unknown")
        with self.assertRaises(QueryError):
            await service.search((session, schema), "A", "怎么用")
        session.call_tool.assert_not_called()

    def test_nested_mcp_auth_error(self):
        error = RuntimeError("private response")
        error.response = SimpleNamespace(status_code=401)
        self.assertEqual(failure_code(ExceptionGroup("transport", [error]), "web"), "mcp_auth_failed")


if __name__ == "__main__":
    unittest.main()
