"""【03】三路公共流程：独占字段、商品级隔离、阶段截止时间和安全日志。

图调用节点实例 → __call__ 校验/超时封装 → 子类 run → products 分发商品任务
→ 子类 retrieve/search 完成外部调用 → 收集候选和诊断 → 返回本路两个字段。
例：向量路负责 embedding_chunks/vector_search_meta，不写网络字段或原有 answer。
这使三个分支可以同时更新不同字段，而不发生整个 state 被反复覆盖的问题。
"""
import asyncio
import logging
import time
from knowledge.processor.query_processor.exceptions import QueryError


def summarize(records, hits):
    """把一条检索路的商品执行记录归纳为状态，不判断候选内容是否正确。

    records 示例：[{"status":"success"}, {"status":"failed"}] → partial。
    全部 failed → failed；全部完成且 hits=[] → empty；完成且有候选 → success。
    超时处理也可追加一条没有 item_index 的节点失败记录，表示整路没有执行完整。
    records=[] 时不会误判“全部失败”，而由 hits 决定 success/empty。
    """
    failed = sum(r["status"] == "failed" for r in records)
    if failed == len(records) and records:
        return "failed"
    if failed:
        return "partial"
    return "success" if hits else "empty"


def failure_code(exc, route):
    """把异常转换为可公开的稳定错误码，保留诊断含义而不暴露响应正文。

    例：QueryError("embedding_failed", ...) 原样取 code；TimeoutError 映射为
    retrieval_timeout；网络异常组中包含 HTTP 401 则映射 mcp_auth_failed。
    MCP 使用异步任务组，真正 HTTP 错误可能在 exceptions 或 __cause__ 里；
    pending 是待检查异常栈，seen 按对象身份避免异常链循环导致无限遍历。
    未识别异常返回“路名_unavailable”，不根据可能含密钥的异常文本进行匹配。
    """
    if isinstance(exc, QueryError):
        return exc.code
    if isinstance(exc, TimeoutError):
        return "retrieval_timeout"
    if route == "web":
        pending, seen = [exc], set()
        while pending:
            current = pending.pop()
            if id(current) in seen:
                continue
            seen.add(id(current))
            status = getattr(getattr(current, "response", None), "status_code", None)
            if status in (401, 403):
                return "mcp_auth_failed"
            if status == 429:
                return "mcp_rate_limited"
            pending.extend(getattr(current, "exceptions", ()))
            if current.__cause__ is not None:
                pending.append(current.__cause__)
    return route + "_unavailable"


class RetrievalNode:
    """三个业务节点共同使用的生命周期，不负责具体编码/检索算法。

    子类声明 route/output/meta，并实现 run。例：VectorSearchNode.route="vector"，
    其预算读 config.vector_timeout，日志名 query.vector，输出字段独占。
    禁用开关只为 HyDE/网络配置；vector_enabled 不存在时 getattr 默认 True。
    """
    route = ""
    output = ""
    meta = ""

    def __init__(self, config):
        """保存配置并取得本路 logger，构造本身不访问外部服务。"""
        self.config = config
        self.logger = logging.getLogger("query." + self.route)

    async def __call__(self, state):
        """【03.1】执行单路并生成新结果，异常路径也必须覆盖旧列表。

        输入：state.retrieval_input={"items":["A","B"],"query":"怎么用？"}。
        输出例：{"embedding_chunks":[A的切片], "vector_search_meta":{
        "status":"partial", "items":[A成功记录,B失败记录], "warnings":[],
        "elapsed_seconds":耗时}}；具体字段由子类 output/meta 决定。
        开关关闭时 skipped；正常零命中 empty；整路失败 failed。
        asyncio.timeout 覆盖整个 run（包括资源等待），到期追加失败记录；已收集
        的商品候选保留。捕获 Exception 不包含 CancelledError，因此请求方取消
        会继续向上传播，不伪装成普通失败结果。耗时使用单调时钟避免系统校时影响。
        """
        start = time.monotonic()
        records, hits, warnings = [], [], []
        status = "failed"
        request_id = state.get("request_id", "")
        self.logger.info("检索开始: request_id=%s, route=%s", request_id, self.route)
        try:
            data = state["retrieval_input"]
            # 即使有人在内部直接调用节点，也执行输入校验，不能只依赖图的 prepare。
            if (not isinstance(data.get("items"), list) or not 1 <= len(data["items"]) <= 5
                    or any(not isinstance(n, str) or not n.strip() or len(n) > 1024 for n in data["items"])
                    or not isinstance(data.get("query"), str) or not data["query"].strip() or len(data["query"]) > 2000):
                raise QueryError("invalid_retrieval_input", "检索输入无效")
            if not getattr(self.config, self.route + "_enabled", True):
                status = "skipped"
            else:
                # timeout 取消仍在执行的子任务；已经完成的商品结果保留供 partial 返回。
                async with asyncio.timeout(getattr(self.config, self.route + "_timeout")):
                    await self.run(data, records, hits, warnings)
                status = summarize(records, hits)
        except Exception as exc:
            code = failure_code(exc, self.route)
            records.append(dict(status="failed", error_code=code))
            status = summarize(records, hits)
            self.logger.error("检索失败: request_id=%s, route=%s, error_code=%s, error_type=%s",
                              request_id, self.route, code, type(exc).__name__)
        self.logger.info("检索结束: request_id=%s, route=%s, status=%s, count=%d", request_id, self.route, status, len(hits))
        # 只返回状态增量；不能 return state，否则三路会同时写入彼此的字段。
        return {self.output: hits, self.meta: dict(status=status, items=records,
                    warnings=sorted(set(warnings)), elapsed_seconds=time.monotonic()-start)}

    async def products(self, data, records, hits, warnings, operation):
        """【03.2】把一条路拆成商品任务，逐个收集，不让单件失败拖垮整路。

        operation(item, query) 是子类提供的异步函数，返回 (候选列表, 告警列表)。
        records/hits/warnings 是 __call__ 创建的本路容器，此函数原地追加，无返回值。
        例：items=[A,B,C] 时先准许两件运行，空出名额后执行第三件；B先完成可先
        保存结果，不必等待A。正常结束后按输入商品顺序和各自 rank 稳定排序。
        路超时时 finally 取消未完成任务，收集过的结果仍保留；异常退出不经过末尾
        排序，因此超时部分结果可能按完成顺序排列，不能假设它们是跨商品排名。
        """
        semaphore = asyncio.Semaphore(2)

        async def one(index, item):
            """一件商品的隔离边界：返回序号、候选、告警和状态四元组。

            例：A正常无命中 → (0, [], [], {item_index:0,status:"empty",...})；
            A检索异常 → failed 记录，其他商品任务照常执行。
            """
            async with semaphore:
                try:
                    found, notices = await operation(item, data["query"])
                    return index, found, notices, dict(item_index=index, status="success" if found else "empty", error_code="")
                except Exception as exc:
                    code = failure_code(exc, self.route)
                    self.logger.warning("商品检索失败: request_id=%s, route=%s, item_index=%d, error_code=%s",
                                        data.get("request_id", ""), self.route, index, code)
                    return index, [], [], dict(item_index=index, status="failed", error_code=code)

        # create_task 提交并发任务，信号量限制实际进入 operation 的数量；
        # as_completed 按完成顺序交回结果，使慢商品不阻止保存已完成商品的证据。
        tasks = [asyncio.create_task(one(i, item)) for i, item in enumerate(data["items"])]
        try:
            for task in asyncio.as_completed(tasks):
                _, found, notices, record = await task
                records.append(record)
                hits.extend(found)
                warnings.extend(notices)
        finally:
            # cancel 是取消请求，不是同步终止；gather 等待协程清理资源。
            # return_exceptions=True 只用于收尾，避免清理阶段子任务异常遮蔽原异常。
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        records.sort(key=lambda r: r["item_index"])
        # 并发完成顺序不稳定；同商品按服务端排名稳定输出，分数不跨商品比较。
        order = {item: i for i, item in enumerate(data["items"])}
        hits.sort(key=lambda h: (order.get(h.get("item_name") or (h.get("item_names") or [""])[0], 0), h.get("rank", 0)))
