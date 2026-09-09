"""【06】商品名只读仓储：一条名称的双向量 → Milvus 记录及融合分数。

search(vectors) → 首次校验/加载集合 → dense(COSINE) + sparse(IP) 两路 ANN
→ WeightedRanker 等权融合 → 解析命中 → 必要时扩召回 → ItemSearchResult。
这里只读商品名集合，不检索正文切片、不判断 confirmed，也不自动创建缺失的集合。
"""

import json
import logging
import threading
import time

from pymilvus import AnnSearchRequest, DataType, WeightedRanker
from pydantic import ValidationError

from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.schema.query_models import ItemHit, ItemSearchResult
from knowledge.service.embedding_vector_util import validate_vectors
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger("query.item_name_search")


class ItemNameSearchRepository:
    """把 Milvus 字段、索引和返回格式细节封装起来，供节点直接读取 ItemHit。"""
    def __init__(self, config, client=None):
        """可注入测试客户端；就绪状态只在当前仓储实例中缓存。"""
        self.config = config
        self.client = client
        self._ready_client = None
        self._lock = threading.Lock()

    def _validate_collection(self, client):
        """【06.1】先校验 schema 和索引，再 load_collection 供向量检索使用。

        例如集合是 768 维而当前模型输出 1024 维，直接报 schema_incompatible。
        集合缺失报 knowledge_not_ready；不能创建空集合后误导用户“商品不存在”。
        """
        cfg = self.config.shared
        args = dict(collection_name=cfg.item_name_collection, timeout=cfg.milvus_timeout_seconds)
        if not client.has_collection(**args):
            raise QueryError("knowledge_not_ready", "商品名称知识库尚未创建，请先完成导入")
        schema = client.describe_collection(**args)
        # 把字段列表转换为“字段名→定义”的字典，后续检查不依赖服务端返回字段顺序。
        fields = {f["name"]: f for f in schema.get("fields", [])}
        expected = {"pk": (DataType.VARCHAR, 64), "document_id": (DataType.VARCHAR, 128),
                    "item_name": (DataType.VARCHAR, 1024), "file_title": (DataType.VARCHAR, 1024),
                    "embedding_model": (DataType.VARCHAR, 128), "updated_at": (DataType.INT64, None),
                    "dense_vector": (DataType.FLOAT_VECTOR, None),
                    "sparse_vector": (DataType.SPARSE_FLOAT_VECTOR, None)}
        valid = schema.get("auto_id") is False and not schema.get("enable_dynamic_field", False)
        for name, (kind, length) in expected.items():
            field = fields.get(name, {})
            valid = valid and field.get("type") == kind
            if length is not None:
                valid = valid and str(field.get("params", {}).get("max_length")) == str(length)
        valid = valid and fields.get("pk", {}).get("is_primary") is True
        valid = valid and str(fields.get("dense_vector", {}).get("params", {}).get("dim")) == str(cfg.embedding_dim)
        if not valid:
            raise QueryError("schema_incompatible", "商品名集合字段、主键或向量维度不兼容")
        # schema 正确仍可能配错索引度量；COSINE/IP 决定后面的融合分数刻度，必须一致。
        indexes = client.list_indexes(**args)
        for name, field, metric in (("dense_vector_index", "dense_vector", "COSINE"),
                                    ("sparse_vector_index", "sparse_vector", "IP")):
            if name not in indexes:
                raise QueryError("schema_incompatible", "商品名集合缺少混合检索索引")
            info = client.describe_index(**args, index_name=name)
            actual_metric = info.get("metric_type") or info.get("params", {}).get("metric_type")
            if str(actual_metric).upper() != metric or info.get("field_name", field) != field:
                raise QueryError("schema_incompatible", "商品名集合索引度量不兼容")
        client.load_collection(**args)
        logger.info("商品名查询集合就绪: collection=%s, dim=%d", cfg.item_name_collection, cfg.embedding_dim)

    def search(self, vectors) -> ItemSearchResult:
        """单个商品表述执行一次混合检索，返回命中记录、截断标志和诊断信息。

        输出示意：hits=[ItemHit(item_name="RS-12 数字万用表", score=0.83, ...)]。
        同一商品可能因有多份说明书返回多条记录，后续 align 才负责名称分组。
        """
        cfg = self.config.shared
        started = time.perf_counter()
        try:
            dense, sparse = validate_vectors(vectors.dense, vectors.sparse, cfg.embedding_dim)
        except Exception as exc:
            raise QueryError("embedding_failed", "查询向量格式不合法", exc) from exc
        try:
            client = self.client if self.client is not None else StorageClients.get_milvus(cfg)
            # 只串行首次校验/加载；正常搜索放在锁外，避免将所有查询串行化。
            # “is”比较客户端对象身份；换了连接对象后重新检查集合。
            with self._lock:
                if self._ready_client is not client:
                    self._validate_collection(client)
                    self._ready_client = client
            args = dict(collection_name=cfg.item_name_collection, timeout=cfg.milvus_timeout_seconds,
                        consistency_level="Strong")
            # Strong 用于本次读的可见性，不是商品集合与正文切片集合之间的事务。
            # 【06.2】两路使用同一模型过滤条件，防止只因向量维度相同就混查不同模型。
            # json.dumps 只用于 Milvus 字符串字面量转义，不作为 shell 转义使用。
            literal = json.dumps(cfg.bge_m3_model_name, ensure_ascii=True)
            expression = f"embedding_model == {literal}"
            incompatible = client.query(**args, filter=f"embedding_model != {literal}",
                                        output_fields=["pk"], limit=1)
            warnings = ["incompatible_embedding_model"] if incompatible else []
            limit = self.config.item_name_top_k
            while True:
                # 【06.3】data=[dense]/[sparse] 的外层列表表示“一个查询向量”。
                # 两个 AnnSearchRequest 表示两条召回通道，不表示查询了两个商品。
                # 两路的 limit 是各自召回记录数；hybrid_search 的 limit 是融合后返回数。
                reqs = [AnnSearchRequest(data=[dense], anns_field="dense_vector",
                                         param={"metric_type": "COSINE"}, limit=limit, expr=expression),
                        AnnSearchRequest(data=[sparse], anns_field="sparse_vector",
                                         param={"metric_type": "IP"}, limit=limit, expr=expression)]
                # 【06.4】权重顺序必须对应 reqs 顺序：先 dense，后 sparse。
                # 本项目已验收的 Milvus 3.0.0 中，同一记录两路命中时：
                # dense_norm=(1+cosine)/2，sparse_norm=0.5+atan(ip)/π，
                # score=0.5*dense_norm+0.5*sparse_norm。
                # 如 cosine=0.8、ip=1，score=0.825；单路未召回则那一路贡献为零。
                # norm_score 由服务端执行，客户端不再次归一化；round_decimal=-1 不预先舍入。
                raw = client.hybrid_search(**args, reqs=reqs,
                    ranker=WeightedRanker(self.config.item_name_dense_weight,
                                          self.config.item_name_sparse_weight, norm_score=True),
                    limit=limit, output_fields=["pk", "document_id", "item_name", "file_title", "embedding_model"],
                    round_decimal=-1)
                hits = self._parse(raw)
                if not hits and incompatible:
                    # 只有其他模型的记录 → 当前模型知识库未就绪；合法空集合则可正常无匹配。
                    compatible = client.query(**args, filter=expression, output_fields=["pk"], limit=1)
                    if not compatible:
                        raise QueryError("knowledge_not_ready", "知识库尚无当前向量模型的商品名称记录")
                # 【06.5】若末位仍可能参与决策，就继续扩召回，避免文档副本遮住竞争产品。
                # 例：50 条全是某商品的说明书、末位分数 0.8，后面可能还有其他商品，扩大到 100。
                # 默认最多 200；仍满额则返回 truncated=True，让 align 澄清而不是猜“唯一商品”。
                saturated = len(hits) >= limit and hits[-1].score >= self.config.item_name_mid_confidence
                if not saturated or limit >= self.config.item_name_max_top_k:
                    break
                limit = min(limit * 2, self.config.item_name_max_top_k)
                logger.info("扩大商品名候选召回: top_k=%d", limit)
            if saturated:
                warnings.append("candidate_recall_truncated")
            if warnings:
                logger.warning("商品名检索存在诊断标识: warnings=%s", ",".join(warnings))
            logger.info("商品名混合检索完成: hits=%d, top_k=%d, weights=0.5/0.5, elapsed=%.3fs",
                        len(hits), limit, time.perf_counter() - started)
            return ItemSearchResult(hits=hits, truncated=saturated, warnings=warnings)
        except QueryError:
            raise
        except Exception as exc:
            # 网络恢复后重做检查和加载，不能永远缓存失效的就绪状态。
            self._ready_client = None
            raise QueryError("milvus_unavailable", "商品名称检索服务暂时不可用", exc) from exc

    def _parse(self, raw):
        """【06.6】把 Milvus 双层列表转为已校验、按高分优先排列的 ItemHit 列表。

        raw 示意：[[{"entity": {"pk": "p1", "item_name": "RS-12", ...}, "distance": 0.83}]]。
        外层对应查询，本函数只接收一个查询，因此长度必须为 1；[[]] 是合法无命中。
        hybrid_search 返回字段名虽叫 distance，这里的含义是越大越好的融合 score。
        """
        try:
            if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], list):
                raise ValueError("单查询必须返回一组命中")
            hits = []
            seen = set()
            for row in raw[0]:
                entity = row["entity"]
                if not isinstance(entity, dict):
                    raise ValueError("命中实体必须是字段对象")
                # **entity 展开标量字段，随后明确映射主键和融合分数。
                # 验证失败（缺字段、NaN、分数超范围）整体报错，不能跳过坏记录再继续确认。
                hit = ItemHit.model_validate({**entity, "pk": entity.get("pk", row.get("id")),
                                              "score": row["distance"]})
                if hit.embedding_model != self.config.shared.bge_m3_model_name or hit.pk in seen:
                    raise ValueError("检索模型或主键不符合契约")
                seen.add(hit.pk)
                hits.append(hit)
            # -score 实现降序；同分时按名称、主键稳定排序，便于复现候选展示。
            return sorted(hits, key=lambda h: (-h.score, h.item_name, h.pk))
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            raise QueryError("invalid_search_result", "商品名检索结果不符合评分或字段契约", exc) from exc
