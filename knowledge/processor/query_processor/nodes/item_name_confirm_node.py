"""【03】商品名确认的业务总流程，建议先读 _process 再进入各 service。

示例问题：“RS-12 怎么测电阻？”
  04 extractor：提取 QueryMention(name="RS-12", evidence="RS-12", ...)
  05 embedding：把提取名编码成一组 dense/sparse，不编码整段问题
  06 repository：两路 ANN 融合，返回库内名称及 score
  07 aligner.align：决定这个表述是否对应某个库内商品
  08 aligner.aggregate：决定整个问题是否能发布商品范围
下例中的名称与分数仅帮助理解；真实输出由模型及当前知识库决定。
"""

import logging

from pydantic import ValidationError

from knowledge.processor.query_processor.base import QueryBaseNode
from knowledge.processor.query_processor.config import QueryConfig
from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.processor.query_processor.state import empty_result
from knowledge.schema.query_models import QueryItemExtraction
from knowledge.service.item_name_aligner import GENERIC_NAMES, ItemNameAligner, normalize_name
from knowledge.service.item_name_extractor import ItemNameExtractor
from knowledge.service.item_name_search_repository import ItemNameSearchRepository
from knowledge.service.query_embedding_service import QueryEmbeddingService

logger = logging.getLogger("query.item_name_confirm")


class ItemNameConfirmNode(QueryBaseNode):
    """只负责执行顺序；提取、向量化、读库和业务评分分别交给独立服务。"""
    name = "item_name_confirm"

    def __init__(self, config=None, *, extractor=None, embedding=None, repository=None):
        """初始化服务，不立刻加载模型或访问数据库。

        测试可注入假 extractor/embedding/repository，让评分分支测试无需外部服务。
        使用 is not None 区分“没有传依赖”和“传入了自定义依赖”。
        """
        self.config = config or QueryConfig()
        self.extractor = extractor if extractor is not None else ItemNameExtractor(self.config)
        self.embedding = embedding if embedding is not None else QueryEmbeddingService(self.config)
        self.repository = repository if repository is not None else ItemNameSearchRepository(self.config)
        self.aligner = ItemNameAligner(self.config)

    def _process(self, request):
        """接收基类已校验的 QueryInput，最后返回聚合后的状态更新。

        mentions 是“用户提到的表述”；item_names 才是最终可供下游使用的库内名称。
        例如 mentions=["RS-12"] 最终可能得到 item_names=["RS PRO RS-12 数字万用表"]。
        """
        # 【03.1 → 04】结构化提取。LLM 只负责提取表述，不能直接宣布商品确认成功。
        try:
            extracted = QueryItemExtraction.model_validate(self.extractor.extract(request))
        except ValidationError as exc:
            raise QueryError("llm_invalid_output", "商品名称识别结果格式不合法", exc) from exc
        if extracted.too_many_products:
            # 超过五个目标时整体反问拆分，不能只检索前五个然后声称回答了全部问题。
            result = empty_result()
            result.update(item_confirm_status="needs_clarification", answer="一次最多确认五个商品，请拆分问题。",
                          warnings=["too_many_products"])
            return result
        if len(extracted.mentions) > self.config.item_name_max_extracted:
            raise QueryError("invalid_input", "一次最多确认五个商品，请拆分问题")
        mentions = []
        seen = set()
        # 【03.2】set 负责快速查重，list 保留首次出现顺序。
        # 如模型重复提取 "RS-12"、"rs-12"，只为规范化后的同一表述编码一次。
        for mention in extracted.mentions:
            key = normalize_name(mention.name)
            if key not in seen:
                seen.add(key)
                mentions.append(mention)
        # 【03.3 → 05】纯品类词暂不检索，但保留在 mentions 中，后面必须提示补充型号。
        # 例如 ["RS-12", "电脑"] 只编码 RS-12；不能直接删掉“电脑”而放行整个问题。
        searchable = [m for m in mentions if normalize_name(m.name) not in GENERIC_NAMES]
        vectors = self.embedding.embed([m.name for m in searchable]) if searchable else []
        if len(vectors) != len(searchable):
            raise QueryError("embedding_failed", "查询向量数量与商品表述不一致")
        # zip 按顺序配对名称和向量；strict=True 禁止两边长度不同时悄悄截断。
        # 映射示意：{"rs-12": EmbeddingVectors(...), "ut61e": EmbeddingVectors(...)}。
        vector_by_name = dict(zip((normalize_name(m.name) for m in searchable), vectors, strict=True))
        results = []
        for index, mention in enumerate(mentions):
            # 【03.4 → 06 / 07】逐表述检索和评分，m1/m2 用于日志及候选分组。
            # 任何一次外部调用失败都向基类抛异常，基类会清空本次部分成功结果。
            key = normalize_name(mention.name)
            if key in GENERIC_NAMES:
                results.append(dict(mention_id=f"m{index + 1}", **mention.model_dump(), status="needs_clarification",
                                    reason="generic_name", confirmed=None, candidates=[], warnings=[]))
                continue
            matched = self.repository.search(vector_by_name[key])
            result = self.aligner.align(mention, matched, f"m{index + 1}")
            results.append(result)
            logger.info("商品表述决策: request_id=%s, mention_id=%s, groups=%d, reason=%s",
                        request.request_id, result["mention_id"], len(result["candidates"]), result["reason"])
        # 【03.5 → 08】只在这里决定整个问题的最终状态。
        # A 已确认、B 未确认 → 整体澄清；A/B 都确认 → 发布两个商品名。
        # mentions 为空时 results=[]，aggregate 会返回“请提供具体商品名称或型号”。
        return self.aligner.aggregate(results, request.original_query)
