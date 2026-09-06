"""商品名称的本地 BGE-M3 混合向量服务。

例如输入“RS PRO RS-12 数字万用表”，模型会同时返回 1024 维 dense 向量和类似
``{12: 0.7, 98: 0.2}`` 的 sparse 权重。前者表达整体语义，后者保留型号、品牌等
关键词信号；两种向量会写入同一条 Milvus 实体，供后续混合检索使用。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import EmbeddingError
from knowledge.utils.client.ai_clients import AIClients


logger = logging.getLogger("import.item_name_embedding")


@dataclass(frozen=True, slots=True)
class EmbeddingVectors:
    """Milvus 一条实体所需的稠密和稀疏向量。

    ``frozen`` 防止校验完成后再被修改，``slots`` 避免为这种高频小对象创建动态属性表。
    """

    dense: list[float]
    sparse: dict[int, float]


class ItemNameEmbeddingService:
    """封装 BGE-M3 返回格式兼容与向量质量校验。"""

    def __init__(self, config: ImportConfig) -> None:
        self.config = config

    def embed(self, item_name: str) -> EmbeddingVectors:
        """为单个商品名称生成可直接写入 Milvus 的混合向量。

        此处调用的是进程内模型，不是远程 embedding HTTP 接口；首次运行若本机无缓存，
        底层模型加载器会先下载 ``BGE_M3_MODEL_NAME`` 对应的权重。
        """
        client = AIClients.get_bge_m3(self.config)
        started_at = time.perf_counter()
        try:
            result = client.encode_documents([item_name])
            dense = self._extract_dense_vector(result)
            sparse = self._extract_sparse_vector(result)
            self.validate_vectors(dense, sparse)
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.error(
                "BGE-M3 商品名称向量化失败: item_name=%s, error_type=%s",
                item_name,
                type(exc).__name__,
            )
            raise EmbeddingError(
                message="BGE-M3 商品名称向量化失败",
                node_name="item_name_embedding_service",
                cause=exc,
            ) from exc

        logger.info(
            "BGE-M3 商品名称向量化完成: model=%s, dense_dim=%d, "
            "sparse_nnz=%d, elapsed=%.3fs",
            self.config.bge_m3_model_name,
            len(dense),
            len(sparse),
            time.perf_counter() - started_at,
        )
        return EmbeddingVectors(dense=dense, sparse=sparse)

    @staticmethod
    def _extract_dense_vector(result: Any) -> list[float]:
        """兼容 pymilvus-model 与 FlagEmbedding 的 dense 字段命名。"""
        if not isinstance(result, dict):
            raise EmbeddingError(message="BGE-M3 返回值必须是字典")
        dense_rows = result.get("dense", result.get("dense_vecs"))
        if dense_rows is None or len(dense_rows) != 1:
            raise EmbeddingError(message="BGE-M3 dense 数量与输入不一致")
        row = dense_rows[0]
        values = row.tolist() if hasattr(row, "tolist") else list(row)
        return [float(value) for value in values]

    @staticmethod
    def _extract_sparse_vector(result: Any) -> dict[int, float]:
        """把 CSR 或 lexical_weights 统一为 Milvus 接受的稀疏字典。

        CSR 用 indptr 标记每一行在 indices/data 中的区间。单条文本时取第一行区间，
        例如 indices=[12, 98]、data=[0.7, 0.2] 会变成 {12: 0.7, 98: 0.2}。
        """
        if not isinstance(result, dict):
            raise EmbeddingError(message="BGE-M3 返回值必须是字典")
        sparse_rows = result.get("sparse", result.get("lexical_weights"))
        if sparse_rows is None:
            raise EmbeddingError(message="BGE-M3 未返回 sparse 向量")

        if hasattr(sparse_rows, "indptr"):
            start = int(sparse_rows.indptr[0])
            end = int(sparse_rows.indptr[1])
            token_ids = sparse_rows.indices[start:end].tolist()
            weights = sparse_rows.data[start:end].tolist()
            # indices 是 token ID，data 是权重；颠倒后 Milvus 无法解析。
            pairs = [
                (int(token_id), float(weight))
                for token_id, weight in zip(token_ids, weights, strict=True)
            ]
        else:
            sparse_row = (
                sparse_rows[0]
                if isinstance(sparse_rows, (list, tuple)) and len(sparse_rows) == 1
                else sparse_rows
            )
            if not isinstance(sparse_row, dict):
                raise EmbeddingError(message="BGE-M3 sparse 格式不受支持")
            pairs = [
                (int(token_id), float(weight))
                for token_id, weight in sparse_row.items()
            ]

        if len({token_id for token_id, _ in pairs}) != len(pairs):
            raise EmbeddingError(message="BGE-M3 sparse 含重复 token ID")
        return dict(pairs)

    def validate_vectors(
        self,
        dense: list[float],
        sparse: dict[int, float],
    ) -> None:
        """在进入 Milvus 前验证向量维度和数值范围。

        维度不符通常意味着模型与 Collection Schema 配置不一致；NaN、Inf 或非正稀疏
        权重会导致检索结果不稳定，因此在网络写入之前直接失败并保留可诊断异常。
        """
        if len(dense) != self.config.embedding_dim:
            raise EmbeddingError(
                message=(
                    "BGE-M3 dense 维度不匹配: "
                    f"expected={self.config.embedding_dim}, actual={len(dense)}"
                ),
                node_name="item_name_embedding_service",
            )
        if not all(math.isfinite(value) for value in dense):
            raise EmbeddingError(
                message="BGE-M3 dense 包含 NaN 或 Inf",
                node_name="item_name_embedding_service",
            )
        if not sparse:
            raise EmbeddingError(
                message="BGE-M3 sparse 不能为空",
                node_name="item_name_embedding_service",
            )
        if any(
            token_id < 0 or not math.isfinite(weight) or weight <= 0
            for token_id, weight in sparse.items()
        ):
            raise EmbeddingError(
                message="BGE-M3 sparse 含非法 token ID 或权重",
                node_name="item_name_embedding_service",
            )
