"""【07】一个请求复用一个 MCP 会话，各商品搜索最多并发两次。

基类 __call__ → run 建立会话 → products 调用 service.search → URL去重 →
返回 web_search_docs/web_search_meta。网络输入来自用户问题和标准商品名，
不依赖 HyDE 输出，因此三路能从 prepare 同时开始。
"""
from knowledge.processor.query_processor.retrieval_base import RetrievalNode


class WebMcpSearchNode(RetrievalNode):
    """管理网络路的商品任务与跨商品去重，协议与响应解析留给服务层。"""
    route, output, meta = "web", "web_search_docs", "web_search_meta"

    def __init__(self, config, service):
        """注入百炼服务；关闭网络时基类直接 skipped，不会进入 service.session。"""
        super().__init__(config)
        self.service = service

    async def run(self, data, records, hits, warnings):
        """在同一个已初始化会话中处理所有商品，然后合并重复网址的商品归属。

        例：A、B都返回 https://example.com/manual，最终只保留一份网页，
        item_names=[A,B]；首次保存的标题/摘要/rank保留，不对网页重新评分。
        async with 退出时清理会话，即使商品调用或整个节点超时也会执行清理。
        如果异常导致流程提前退出，末尾跨商品去重尚未执行，已完成结果由基类保留。
        """
        async with self.service.session() as connection:
            async def search(item, query):
                """闭包绑定当前 connection，使 products 仍使用统一的两参数接口。"""
                return await self.service.search(connection, item, query)
            await self.products(data, records, hits, warnings, search)
        # 同网页跨商品出现时合并归属，保留首次证据，避免后续重复引用。
        unique = {}
        for hit in hits:
            if hit["url"] in unique:
                unique[hit["url"]]["item_names"] = sorted(set(unique[hit["url"]]["item_names"] + hit["item_names"]))
            else:
                unique[hit["url"]] = hit
        hits[:] = list(unique.values())
