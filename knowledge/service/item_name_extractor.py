"""【04】从问题及可选历史中提取商品表述，不访问商品数据库。

输入 QueryInput(original_query="RS-12 怎么用？", history=[])；期望结构示例：
  {"mentions": [{"name": "RS-12", "evidence": "RS-12", "source": "current_query"}],
   "rewritten_query": "RS-12 如何使用？", "too_many_products": false}
字段格式和证据通过后返回 QueryItemExtraction，交回节点继续向量化。
有效的 mentions=[] 表示无法提取；模型格式错误则抛异常，两者不能混淆。
"""

import json
import logging
import time
from functools import lru_cache

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.prompt.query_prompt import ITEM_NAME_QUERY_SYSTEM
from knowledge.schema.query_models import QueryInput, QueryItemExtraction

logger = logging.getLogger("query.item_name_extractor")


@lru_cache(maxsize=4)
def _query_llm(model, api_base, api_key, timeout, retries, max_tokens):
    """按参数缓存至多四组查询客户端；相同配置复用实例，不重复建立连接。

    lru_cache 缓存的是客户端对象，不缓存用户问题或模型回复；不要记录其敏感参数。
    """
    # 查询链拥有独立输出预算，不修改导入共用的 300-token 单例。
    from langchain_deepseek import ChatDeepSeek
    return ChatDeepSeek(model=model, base_url=api_base.rstrip("/"), api_key=api_key,
                        temperature=0.0, max_tokens=max_tokens, timeout=timeout,
                        max_retries=retries, extra_body={"thinking": {"type": "disabled"}})


class ItemNameExtractor:
    """负责模型调用、结构化校验和原文证据检查，商品是否存在由检索层判断。"""
    def __init__(self, config: QueryConfig, llm=None):
        self.config = config
        self.llm = llm

    def extract(self, request: QueryInput) -> QueryItemExtraction:
        """正常返回经过证据校验的提取对象；格式/证据失败最多重新生成一次。"""
        started = time.perf_counter()
        try:
            cfg = self.config.shared
            if self.llm is None:
                if not all((cfg.deepseek_llm_model, cfg.deepseek_api_base, cfg.deepseek_api_key)):
                    raise QueryError("llm_unavailable", "商品名称识别服务尚未配置")
                llm = _query_llm(cfg.deepseek_llm_model, cfg.deepseek_api_base,
                                 cfg.deepseek_api_key, cfg.deepseek_timeout_seconds,
                                 cfg.deepseek_max_retries, self.config.llm_max_tokens)
            else:
                llm = self.llm
            # 【04.1】以 Pydantic 模型声明模型必须遵守的结构。
            # function_calling 用于约束输出形态，这里并不是让 LLM 自主操作 Milvus。
            chain = llm.with_structured_output(QueryItemExtraction, method="function_calling")
            # 系统消息放固定规则，用户消息放 JSON 数据；历史不拼进系统指令。
            # model_dump 将 Pydantic HistoryMessage 转成 json.dumps 能处理的普通字典。
            messages = [SystemMessage(content=ITEM_NAME_QUERY_SYSTEM), HumanMessage(content=json.dumps(
                {"query": request.original_query, "history": [m.model_dump() for m in request.history]},
                ensure_ascii=False))]
            for attempt in range(2):
                try:
                    # 【04.2】invoke 才发生真实远程调用；格式正确后仍须检查名称来自原文。
                    result = QueryItemExtraction.model_validate(chain.invoke(messages))
                    self._validate_evidence(result, request)
                    logger.info("商品表述提取完成: request_id=%s, count=%d, elapsed=%.3fs",
                                request.request_id, len(result.mentions), time.perf_counter() - started)
                    return result
                except (ValidationError, OutputParserException, ValueError) as exc:
                    # 【04.3】此循环只处理解析/证据错误；网络重试由底层客户端参数控制。
                    # attempt=0 失败可再生成一次；attempt=1 再失败则上抛，不能返回空商品列表。
                    if attempt:
                        raise QueryError("llm_invalid_output", "商品名称识别结果不符合格式或证据要求", exc) from exc
                    logger.warning("重试商品提取格式校验: request_id=%s, error_type=%s",
                                   request.request_id, type(exc).__name__)
                    # 不把无效响应或异常正文回传，避免泄漏及强化模型幻觉。
                    messages.append(HumanMessage(content="请重新提取，严格遵守结构及原文证据约束，不要补写名称。"))
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError("llm_unavailable", "商品名称识别服务暂时不可用", exc) from exc

    def _validate_evidence(self, result, request):
        """【04.4】校验 source→evidence→name 的来源关系。

        正例：问题“RS-12 怎么用”，evidence="RS-12"、name="RS-12"。
        反例：问题只有“L420”，name 却补成“L420x”，证据中找不到新名称，拒绝通过。
        source=history 只从调用方传入的历史找证据，本函数不读取会话数据库。
        """
        from knowledge.service.item_name_aligner import normalize_name
        if len(result.mentions) > self.config.item_name_max_extracted:
            raise ValueError("提取目标超过限制")
        for mention in result.mentions:
            sources = ([request.original_query] if mention.source == "current_query"
                       else [m.content for m in request.history])
            if not any(mention.evidence in source for source in sources):
                raise ValueError("商品证据不能在原文定位")
            # 名称允许大小写/空白规范化后比较；evidence 本身仍要求是对应来源的原文片段。
            # 这能约束补写名称，不能证明历史内容正确，也不能代替后续商品身份判定。
            if normalize_name(mention.name) not in normalize_name(mention.evidence):
                raise ValueError("商品名称必须来自证据，不能补全型号")
