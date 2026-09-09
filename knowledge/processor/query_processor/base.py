"""查询节点边界统一记录耗时，将技术错误转换为可路由的 error 状态。"""

import logging
import time
import uuid

from pydantic import ValidationError

from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.processor.query_processor.state import empty_result
from knowledge.schema.query_models import QueryInput

logger = logging.getLogger("query.node")


class QueryBaseNode:
    """【02】所有查询节点共用的执行外壳，类似 Java 的模板方法模式。

    LangGraph 调用 node(state) → __call__ → process → 子类的 _process。
    直接 node.process(state) 也经过校验、日志和异常转换；业务代码只写在 _process 中。
    """

    def _request(self, state):
        """把宽松图状态转换成经过校验的 QueryInput，不原地修改调用方 state。"""
        if not isinstance(state, dict):
            raise QueryError("invalid_input", "查询输入必须是对象")
        try:
            # 【02.1】字典推导式只挑 QueryInput 声明的输入字段。
            # 例如 state 中还有 item_names、answer，不能一起传给禁止额外字段的输入模型。
            request = QueryInput.model_validate({k: state[k] for k in QueryInput.model_fields if k in state})
            if (len(request.original_query) > self.config.max_query_chars
                    or len(request.history) > self.config.max_history_messages
                    or sum(len(m.content) for m in request.history) > self.config.max_history_chars):
                raise ValueError("查询或历史长度超限")
        except (ValidationError, ValueError) as exc:
            raise QueryError("invalid_input", "请提供非空问题，并遵守问题和历史长度限制", exc) from exc
        # model_copy 返回新模型；缺省生成 request_id，同一个请求的各步日志使用同一标识。
        return request.model_copy(update={"request_id": request.request_id or uuid.uuid4().hex})

    def process(self, state):
        """校验 → 调用业务 → 统一结束；返回本节点负责字段的更新字典。

        正常示例：_process 返回 confirmed，原样交给图合并。
        失败示例：仓储抛 QueryError("milvus_unavailable", ...)，这里转换成
        {status: error, item_names: [], error_code: milvus_unavailable, ...}。
        实际状态键为 item_confirm_status；错误时不能保留先前成功的商品范围。
        """
        started = time.perf_counter()
        request_id = uuid.uuid4().hex
        try:
            request = self._request(state)
            request_id = request.request_id
            logger.info("商品名确认开始: request_id=%s", request_id)
            # 【02.2 → 03】动态调用 ItemNameConfirmNode._process，开始实际商品确认。
            result = self._process(request)
        except QueryError as exc:
            # 【02.3】已分类的外部错误保留 code；cause 仅用于诊断，不展开敏感异常正文。
            result = empty_result()
            result.update(item_confirm_status="error", error_code=exc.code, answer=str(exc))
            logger.error("商品名确认失败: request_id=%s, code=%s, error_type=%s", request_id,
                         exc.code, type(exc.cause).__name__ if exc.cause else type(exc).__name__)
        except Exception as exc:
            # 未预料异常也收敛到 error，避免下游把“异常中断后的残留状态”当成正常确认。
            result = empty_result()
            result.update(item_confirm_status="error", error_code="internal_error", answer="商品确认暂时失败，请稍后重试。")
            logger.error("商品名确认异常: request_id=%s, error_type=%s", request_id, type(exc).__name__)
        result["request_id"] = request_id
        # 【08 → 09】正常与错误都经过结束日志。返回值由图合并，再由 CLI 输出 JSON。
        logger.info("商品名确认结束: request_id=%s, status=%s, elapsed=%.3fs", request_id,
                    result["item_confirm_status"], time.perf_counter() - started)
        return result

    def __call__(self, state):
        """让实例能像函数一样被图调用：node(state) 等价于 node.__call__(state)。"""
        return self.process(state)
