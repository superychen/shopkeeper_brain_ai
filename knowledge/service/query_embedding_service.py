"""【05】将提取名批量转换为 Milvus 查询所需的两种向量。

调用链：节点 → embed(["RS-12", "UT61E"]) → AIClients.get_bge_m3
→ encode_queries → extract_vectors → [EmbeddingVectors, EmbeddingVectors]。
返回列表保持输入顺序；第一组向量只能配第一条名称，错位会直接查错商品。
"""

import logging
import time

from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.service.embedding_vector_util import extract_vectors
from knowledge.service.item_name_embedding_service import EmbeddingVectors
from knowledge.utils.client.ai_clients import AIClients

logger = logging.getLogger("query.embedding")


class QueryEmbeddingService:
    """复用本地 BGE-M3 模型；不请求远程 embedding API，不更改入库向量处理方式。"""
    def __init__(self, config):
        self.config = config

    def embed(self, names: list[str]) -> list[EmbeddingVectors]:
        """返回每个名称的 dense 列表与 sparse 字典，空输入直接返回空列表。

        结构示意：EmbeddingVectors(dense=[0.1, ...], sparse={12: 0.7, 98: 0.2})。
        dense 实际为配置维度（默认 1024），sparse 的键是 token ID，不是汉字本身；
        数值仅为示例，不是相似度分数，不能拿这些权重直接与 0.7 门槛比较。
        """
        if not names:
            return []
        started = time.perf_counter()
        try:
            model = AIClients.get_bge_m3(self.config.shared)
            # 【05.1】共享客户端可能已由导入流程初始化；防止改配置后仍误用旧模型。
            active_name = getattr(model, "model_name", None)
            if isinstance(active_name, str) and active_name != self.config.shared.bge_m3_model_name:
                raise QueryError("embedding_failed", "进程缓存模型与查询配置不一致，请重启服务")
            # 【05.2】查询侧用 encode_queries；导入侧用 encode_documents。
            # 两边必须使用同一模型表示，不自行对 sparse 归一化或添加查询专用文字前缀。
            encoded = model.encode_queries(names)
            # 【05.3】公共工具兼容模型的 dense/sparse 返回格式（含 CSR 稀疏矩阵），
            # 同时校验行数、维度、有限数值和非空稀疏权重；坏行不能跳过，否则名称会错位。
            rows = extract_vectors(encoded, len(names), self.config.shared.embedding_dim)
        except QueryError:
            raise
        except Exception as exc:
            raise QueryError("embedding_failed", "商品名称向量化失败", exc) from exc
        logger.info("查询编码完成: count=%d, dim=%d, elapsed=%.3fs", len(rows),
                    self.config.shared.embedding_dim, time.perf_counter() - started)
        # 包装为简单的数据对象交回节点，随后作为 repository.search 的参数。
        return [EmbeddingVectors(dense, sparse) for dense, sparse in rows]
