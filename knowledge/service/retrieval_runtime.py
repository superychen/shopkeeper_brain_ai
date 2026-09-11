"""受控线程桥：超时只取消等待，真实工作结束后才释放在途许可。

同步工作：向量节点 → ENCODING.run(编码函数) / MILVUS.run(数据库函数)。
异步外呼：生成服务 → LLM_CALLS.slot；百炼会话 → MCP_SESSIONS.slot。
例：第一条编码还在GPU上执行，即使请求已经超时，第二条也不能立刻占同一许可；
否则连续超时请求会在后台堆出大量真实推理。许可绑定底层任务完成，而非用户等待结束。
这些对象在当前进程内共享；不负责限制其他进程或旧导入流程的执行数量。
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager

from knowledge.processor.query_processor.exceptions import QueryError


class BoundedExecutor:
    """把同步阻塞函数移入有界线程池，并把其结果桥接为可 await 的结果。

    线程数量和许可数量相同，因此提交前必须先获得名额，避免线程池队列无限增长。
    例：workers=1 代表一次只能提交一个编码任务；等待者最多轮询约5秒再报繁忙。
    """
    def __init__(self, workers):
        """workers 同时决定线程数和在途许可数；构造时不会执行模型函数。"""
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retrieval")
        self.slots = threading.BoundedSemaphore(workers)

    async def run(self, function, *args):
        """执行 function(*args)，异步返回其原结果；异常按原类型传给调用方。

        例：await ENCODING.run(embedding.embed, 文本) 得到 (双向量, profile)。
        提交前每50ms尝试拿许可，最多100轮；等待期间可被外层超时取消。
        提交失败立即归还许可；提交成功后由 Future 完成回调归还。
        await 被取消不保证已运行的线程停止，因此不能在此函数 finally 里归还许可。
        """
        # for...else：只有所有轮次都没 break 才进入 else，表示等待期内一直无名额。
        for _ in range(100):
            if self.slots.acquire(blocking=False):
                break
            await asyncio.sleep(0.05)
        else:
            raise QueryError("retrieval_busy", "检索资源繁忙")
        try:
            future = self.pool.submit(function, *args)
        except BaseException:
            self.slots.release()
            raise
        # 回调绑定 concurrent.futures.Future 的真实终态；_ 表示不读取结果对象。
        # wrap_future 只负责把线程 Future 桥接到当前事件循环，不把同步函数变成协程。
        future.add_done_callback(lambda _: self.slots.release())
        return await asyncio.wrap_future(future)


ENCODING = BoundedExecutor(1)
MILVUS = BoundedExecutor(4)


class AsyncGate:
    """跨请求/事件循环共享在途上限，避免每个节点各限两次后仍无限放大外呼。

    使用 threading.BoundedSemaphore 存名额，不把全局名额绑定到某一个事件循环。
    它仅控制进入异步外呼的数量，不创建线程，真实网络等待仍在事件循环中进行。
    例：四个请求已持有LLM许可，第五个请求先等待；空出名额再进行其模型调用。
    """
    def __init__(self, capacity):
        """capacity 为当前进程允许同时持有该资源的调用数。"""
        self.slots = threading.BoundedSemaphore(capacity)

    @asynccontextmanager
    async def slot(self):
        """用于 async with 的许可生命周期，成功/异常/取消退出时均归还。

        例：async with LLM_CALLS.slot(): await llm.ainvoke(...)。
        yield 前取得许可，yield 暂停本生成器并让调用方执行 with 块，离开后执行 finally。
        和同步线程不同，调用方协程退出此块就释放名额；不能据此保证远端服务已停止计费。
        """
        for _ in range(100):
            if self.slots.acquire(blocking=False):
                break
            await asyncio.sleep(.05)
        else:
            raise QueryError("retrieval_busy", "外部检索资源繁忙")
        try:
            yield
        finally:
            self.slots.release()


LLM_CALLS = AsyncGate(4)
MCP_SESSIONS = AsyncGate(4)
