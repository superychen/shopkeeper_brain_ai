"""【05】正文只读仓储：精确商品过滤＋模型空间过滤，再执行双向量融合。

两路本地节点 → search → 首次_validate → 构建双ANN请求 → hybrid_search
→ 解析真实切片 → 检查不兼容profile记录 → 返回 (hits,warnings)。
例：查询RS-12只读RS-12的兼容正文，不扩大到其他商品，也不局限于确认阶段
命中的少数document_id。集合和向量维度来自共享配置，不把示例集合名写死。
"""
import logging
import math
import threading
from pymilvus import AnnSearchRequest, DataType, WeightedRanker

from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.service.embedding_vector_util import validate_vectors
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger("query.chunk_search")
# 只返回后续证据追踪需要的标量/JSON字段；不把两种大向量带进图状态。
# chunk_id是64字符主键，document_id/file_title/title帮助定位原文，metadata保留标题路径等。
FIELDS = ["chunk_id", "document_id", "item_name", "content", "title", "file_title",
          "chunk_index", "embedding_profile", "metadata"]


class ChunkSearchRepository:
    """封装Milvus请求和字段契约；不创建集合、不写入数据、不生成最终答案。"""
    def __init__(self, config, client=None):
        """client可注入测试替身；_ready缓存已经校验的客户端身份，锁保护首次就绪检查。"""
        self.config, self.client = config, client
        self._ready = None
        self._lock = threading.Lock()

    def _validate(self, client, args):
        """检查已有集合的字段、主键、向量维度和索引度量，通过后加载集合。

        args 示例：{collection_name:配置正文集合,timeout:10}；成功无返回值。
        例：当前模型1024维但集合768维 → chunk_schema_mismatch；集合不存在也报错，
        不能创建一个空集合然后告诉用户“查不到资料”。load_collection只加载已有数据。
        fields转成名称字典，使字段校验不依赖服务端字段排列顺序；正文metadata应为JSON。
        """
        if not client.has_collection(**args):
            raise QueryError("chunk_schema_mismatch", "正文集合不存在")
        schema = client.describe_collection(**args)
        fields = {f["name"]: f for f in schema.get("fields", [])}
        expected = {k: DataType.VARCHAR for k in FIELDS if k not in {"chunk_index", "metadata"}}
        expected.update(chunk_index=DataType.INT64, metadata=DataType.JSON,
                        dense_vector=DataType.FLOAT_VECTOR, sparse_vector=DataType.SPARSE_FLOAT_VECTOR)
        if (any(fields.get(k, {}).get("type") != v for k, v in expected.items())
                or not fields.get("chunk_id", {}).get("is_primary")
                or str(fields.get("dense_vector", {}).get("params", {}).get("dim")) != str(self.config.shared.embedding_dim)):
            raise QueryError("chunk_schema_mismatch", "正文集合字段不兼容")
        for field, metric in (("dense_vector", "COSINE"), ("sparse_vector", "IP")):
            # 字段类型正确还不够：索引度量决定分数含义，必须与入库保持一致。
            info = client.describe_index(**args, index_name=field + "_index")
            actual = info.get("metric_type") or info.get("params", {}).get("metric_type")
            if actual != metric or info.get("field_name", field) != field:
                raise QueryError("chunk_schema_mismatch", "正文索引度量不兼容")
        client.load_collection(**args)

    def search(self, vectors, profile, item_name, route):
        """检索一件标准商品的正文，默认返回最多5条真实切片及兼容性告警。

        输入：vectors=EmbeddingVectors，profile=编码服务给出的指纹，
        item_name="RS-12"，route="vector"或"hyde"。
        输出结构示例（值仅演示）：([{chunk_id:64字符,content:真实正文,
        item_name:"RS-12",route:"vector",rank:1,score:0.6,...}], [])。
        0.6仍保留：商品名确认0.7用于身份判断，正文TopK只为后续重排召回候选。
        没有命中且无profile问题返回([],[])；全部是旧模型正文则抛
        embedding_profile_mismatch，不能解释为知识不存在。
        """
        try:
            cfg = self.config
            client = self.client if self.client is not None else StorageClients.get_milvus(cfg.shared)
            args = dict(collection_name=cfg.shared.chunks_collection,
                        timeout=min(cfg.call_timeout, cfg.shared.milvus_timeout_seconds))
            with self._lock:
                # 首次校验/加载串行，实际search在锁外；换客户端后重新验证。
                if self._ready is not client:
                    self._validate(client, args)
                    self._ready = client
            dense, sparse = validate_vectors(vectors.dense, vectors.sparse, cfg.shared.embedding_dim)
            expr = "item_name in {item_names} and embedding_profile == {profile}"
            # 模板与值分离：名称含双引号/反斜杠仍作为一个值，不能拼成表达式片段。
            # 两个ANN都绑定同样的params，避免dense过滤了商品而sparse仍搜索全库。
            params = {"item_names": [item_name], "profile": profile}
            reqs = [AnnSearchRequest(data=[vector], anns_field=field, param={"metric_type": metric},
                                    limit=limit, expr=expr, expr_params=params)
                    for vector, field, metric, limit in ((dense, "dense_vector", "COSINE", cfg.dense_top_k),
                                                        (sparse, "sparse_vector", "IP", cfg.sparse_top_k))]
            # 每个ANN的limit默认50，是通道候选数；下方limit默认5，是融合后的条数。
            # norm_score让服务端按度量先归一化再等权相加；客户端不再归一化或设0.7阈值。
            # Strong保证本次读取可见性，不构成商品名集合与正文集合之间的跨表事务。
            raw = client.hybrid_search(**args, reqs=reqs, ranker=WeightedRanker(.5, .5, norm_score=True),
                                       limit=cfg.chunks_per_item, output_fields=FIELDS,
                                       consistency_level="Strong", round_decimal=-1)
            if (not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list)
                    or len(raw[0]) > cfg.chunks_per_item):
                raise QueryError("invalid_search_result", "正文响应结构错误")
            hits, seen = [], set()
            # raw外层是查询批次，本函数只有一个查询，所以只接受[命中列表]；[[]]合法。
            # 服务端字段叫distance，但在WeightedRanker结果中是越大越好的融合score。
            for row in raw[0]:
                entity = row["entity"]
                score = row["distance"]
                if (not isinstance(entity, dict) or entity.get("item_name") != item_name or entity.get("embedding_profile") != profile
                        or not isinstance(entity.get("chunk_id"), str) or len(entity["chunk_id"]) != 64
                        or not isinstance(entity.get("document_id"), str) or not entity["document_id"]
                        or type(entity.get("chunk_index")) is not int or entity["chunk_index"] < 0
                        or not isinstance(entity.get("metadata"), dict)
                        or not isinstance(entity.get("content"), str) or not entity["content"].strip()
                        or isinstance(score, bool) or not isinstance(score, (int, float))
                        or not math.isfinite(score) or not 0 <= score <= 1):
                    raise QueryError("invalid_search_result", "正文命中不符合范围或字段契约")
                if entity["chunk_id"] in seen:
                    continue
                seen.add(entity["chunk_id"])
                # 白名单字段复制＋补充来源和排名；同路按chunk_id去重，跨路保留交给后续融合。
                hits.append({k: entity.get(k) for k in FIELDS} | dict(route=route, rank=len(hits)+1,
                    score=score, score_type="milvus_weighted_dense_cosine_sparse_ip"))
            warnings = []
            # 有商品正文但无兼容 profile 是配置问题，不能返回“没有知识”。
            incompatible = client.query(**args, filter="item_name in {item_names} and embedding_profile != {profile}",
                                        filter_params=params, output_fields=["chunk_id"], limit=1)
            if incompatible:
                # 例：A有p旧版和p新版 → 只返回新版并告警；A仅有旧版 → 整件商品失败。
                compatible = client.query(**args, filter=expr, filter_params=params, output_fields=["chunk_id"], limit=1)
                if not compatible:
                    raise QueryError("embedding_profile_mismatch", "正文模型空间不匹配")
                warnings.append("incompatible_embedding_profile")
            logger.info("正文混合召回完成: route=%s, count=%d", route, len(hits))
            return hits, warnings
        except QueryError:
            raise
        except Exception as exc:
            # 不打印第三方异常正文；丢弃就绪缓存，下一次调用重新验证连接/集合。
            self._ready = None
            raise QueryError("milvus_unavailable", "正文检索不可用", exc) from exc
