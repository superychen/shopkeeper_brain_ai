"""商品名称的本地 BGE-M3 混合向量服务。

例如输入“RS PRO RS-12 数字万用表”，模型会同时返回 1024 维 dense 向量和类似
``{12: 0.7, 98: 0.2}`` 的 sparse 权重。前者表达整体语义，后者保留型号、品牌等
关键词信号；两种向量会写入同一条 Milvus 实体，供后续混合检索使用。

调用走向：ItemNameRecognitionNode → embed → AIClients.get_bge_m3
→ encode_documents → extract_vectors → EmbeddingVectors → ItemNameRepository。
这里只编码商品名；整篇文档的切片批量编码由 ChunkEmbeddingService 负责。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import EmbeddingError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.service.embedding_vector_util import extract_vectors, validate_vectors


logger = logging.getLogger("import.item_name_embedding")


@dataclass(frozen=True, slots=True)
class EmbeddingVectors:
    """Milvus 一条实体所需的稠密和稀疏向量。

    ``frozen`` 禁止重新赋值属性，但内部 list/dict 仍可变，使用者应按只读结果处理；
    ``slots`` 避免为这种高频小对象创建动态属性表。
    """

    dense: list[float]
    sparse: dict[int, float]


class ItemNameEmbeddingService:
    """封装 BGE-M3 返回格式兼容与向量质量校验。"""

    def __init__(self, config: ImportConfig) -> None:
        """保存配置；实际模型按需获取，初始化服务时不加载权重。"""
        self.config = config

    def embed(self, item_name: str) -> EmbeddingVectors:
        """为单个商品名称生成可直接写入 Milvus 的混合向量。

        此处调用的是进程内模型，不是远程 embedding HTTP 接口；首次运行若本机无缓存，
        底层模型加载器会先下载 ``BGE_M3_MODEL_NAME`` 对应的权重。
        """
        client = AIClients.get_bge_m3(self.config)
        started_at = time.perf_counter()
        try:
            # 编码接口接收列表，即使只有一个商品名，也要包成单元素批次。
            result = client.encode_documents([item_name])
            # 统一校验后取第 0 条；expected_rows=1 保证不会错取多条响应中的一条。
            dense, sparse = extract_vectors(result, 1, self.config.embedding_dim)[0]
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

    def validate_vectors(self, dense, sparse) -> None:
        """复用公共数值校验；成功无返回值，非法向量抛 EmbeddingError。"""
        validate_vectors(dense, sparse, self.config.embedding_dim)
