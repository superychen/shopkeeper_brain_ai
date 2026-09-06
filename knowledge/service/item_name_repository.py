"""商品名称混合向量的 Milvus 仓储。

该层只处理持久化，不关心商品名称怎样识别。示例：同一 document_id 第一次识别为
“RS PRO RS-12 数字万用表”，重新导入后名称修正，两个请求会生成相同主键并执行
upsert，因此 Milvus 中仍然只有一条该文档的商品记录。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

from pymilvus import DataType, MilvusClient

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import ConfigurationError, MilvusError
from knowledge.service.item_name_embedding_service import EmbeddingVectors
from knowledge.utils.client.storage_clients import StorageClients


logger = logging.getLogger("import.item_name_repository")


class ItemNameRepository:
    """负责集合治理和基于 document_id 的幂等 upsert。"""

    # 线程锁只保护“检查并建表”的临界区，向量计算和正常 upsert 不会被串行化。
    # 它解决同一进程多个导入任务首次写库时同时发现集合不存在的竞争问题。
    _collection_lock = threading.Lock()

    def __init__(self, config: ImportConfig) -> None:
        self.config = config

    def upsert(
        self,
        *,
        collection_name: str,
        document_id: str,
        file_title: str,
        item_name: str,
        vectors: EmbeddingVectors,
    ) -> str:
        """确保集合兼容后写入商品名称，并返回稳定主键。

        主键不使用商品名称，因为名称可能被模型修正；只要 document_id 不变，重试和
        重复导入都会覆盖同一实体。例如 document_id="doc-1" 始终得到同一个 SHA-256。
        """
        client = StorageClients.get_milvus(self.config)
        self.ensure_collection(client, collection_name)
        # 版本前缀为以后调整主键语义留出迁移空间，NUL 分隔符避免字符串拼接歧义。
        item_pk = hashlib.sha256(
            f"item-name:v1\0{document_id}".encode("utf-8")
        ).hexdigest()
        entity = {
            "pk": item_pk,
            "document_id": document_id,
            "file_title": file_title,
            "item_name": item_name,
            "embedding_model": self.config.bge_m3_model_name,
            "updated_at": int(time.time() * 1000),
            "dense_vector": vectors.dense,
            "sparse_vector": vectors.sparse,
        }

        started_at = time.perf_counter()
        try:
            client.upsert(collection_name=collection_name, data=[entity])
        except Exception as exc:
            logger.error(
                "Milvus 商品名称 upsert 失败: collection=%s, pk=%s, error_type=%s",
                collection_name,
                item_pk[:12],
                type(exc).__name__,
            )
            raise MilvusError(
                message="Milvus 商品名称 upsert 失败",
                node_name="item_name_repository",
                cause=exc,
            ) from exc

        logger.info(
            "Milvus 商品名称 upsert 完成: collection=%s, pk=%s, elapsed=%.3fs",
            collection_name,
            item_pk[:12],
            time.perf_counter() - started_at,
        )
        return item_pk

    def ensure_collection(
        self,
        client: MilvusClient,
        collection_name: str,
    ) -> None:
        """幂等创建集合，并拒绝复用不兼容的既有 Schema。

        已存在集合也必须校验，而不是直接复用。例如历史集合的 dense_vector 为 768 维，
        当前 BGE-M3 输出 1024 维时应在 upsert 前报配置错误，而不是写到一半才失败。
        """
        with self._collection_lock:
            try:
                exists = client.has_collection(collection_name=collection_name)
            except Exception as exc:
                raise MilvusError(
                    message="检查 Milvus 商品名称集合失败",
                    node_name="item_name_repository",
                    cause=exc,
                ) from exc

            if not exists:
                try:
                    self.create_collection(client, collection_name)
                except Exception as exc:
                    # 多进程可能同时建表；仅在复查后仍不存在时报告失败。
                    try:
                        concurrently_created = client.has_collection(
                            collection_name=collection_name
                        )
                    except Exception:
                        concurrently_created = False
                    if not concurrently_created:
                        if isinstance(exc, (ConfigurationError, MilvusError)):
                            raise
                        raise MilvusError(
                            message="创建 Milvus 商品名称集合失败",
                            node_name="item_name_repository",
                            cause=exc,
                        ) from exc

            self._validate_schema(client, collection_name)
            self._validate_indexes(client, collection_name)

    def create_collection(
        self,
        client: MilvusClient,
        collection_name: str,
    ) -> None:
        """创建固定字段集合和 dense/sparse 两个索引。

        dense 使用 COSINE 比较整体语义相似度；sparse 使用 IP 累加重合关键词权重，
        例如精确型号 ``RS-12`` 通常能从 sparse 通道获得更强的匹配信号。
        """
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="pk", datatype=DataType.VARCHAR, is_primary=True, max_length=64
        )
        schema.add_field(
            field_name="document_id", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(
            field_name="file_title", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="item_name", datatype=DataType.VARCHAR, max_length=1024
        )
        schema.add_field(
            field_name="embedding_model", datatype=DataType.VARCHAR, max_length=128
        )
        schema.add_field(field_name="updated_at", datatype=DataType.INT64)
        schema.add_field(
            field_name="dense_vector",
            datatype=DataType.FLOAT_VECTOR,
            dim=self.config.embedding_dim,
        )
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_name="dense_vector_index",
            index_type="AUTOINDEX",
            metric_type="COSINE",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_vector_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info("Milvus 商品名称集合创建完成: collection=%s", collection_name)

    def _validate_schema(self, client: MilvusClient, collection_name: str) -> None:
        """校验关键字段类型、长度、主键策略和 dense 维度。

        这里把服务端返回的 fields 转成按名称索引的字典，后续校验不依赖字段顺序。
        这比逐下标比较更稳健，也便于错误信息直接指出不兼容字段。
        """
        try:
            description = client.describe_collection(collection_name=collection_name)
        except Exception as exc:
            raise MilvusError(
                message="读取 Milvus 商品名称集合 Schema 失败",
                node_name="item_name_repository",
                cause=exc,
            ) from exc

        if description.get("auto_id") is not False:
            self._incompatible("商品名称集合必须关闭 auto_id")
        if description.get("enable_dynamic_field") is True:
            self._incompatible("商品名称集合必须关闭动态字段")

        fields = {
            field.get("name"): field
            for field in description.get("fields", [])
            if isinstance(field, dict) and field.get("name")
        }
        expected_types = {
            "pk": DataType.VARCHAR,
            "document_id": DataType.VARCHAR,
            "file_title": DataType.VARCHAR,
            "item_name": DataType.VARCHAR,
            "embedding_model": DataType.VARCHAR,
            "updated_at": DataType.INT64,
            "dense_vector": DataType.FLOAT_VECTOR,
            "sparse_vector": DataType.SPARSE_FLOAT_VECTOR,
        }
        for field_name, expected_type in expected_types.items():
            field = fields.get(field_name)
            if field is None or int(field.get("type", -1)) != int(expected_type):
                self._incompatible(f"商品名称集合字段不兼容: {field_name}")
        if fields["pk"].get("is_primary") is not True:
            self._incompatible("商品名称集合 pk 必须是主键")

        expected_lengths = {
            "pk": 64,
            "document_id": 128,
            "file_title": 1024,
            "item_name": 1024,
            "embedding_model": 128,
        }
        for field_name, expected_length in expected_lengths.items():
            try:
                actual_length = int(fields[field_name].get("params", {}).get("max_length"))
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    message=f"商品名称集合字段缺少 max_length: {field_name}",
                    node_name="item_name_repository",
                    cause=exc,
                ) from exc
            if actual_length != expected_length:
                self._incompatible(f"商品名称集合字段长度不兼容: {field_name}")

        try:
            actual_dim = int(fields["dense_vector"].get("params", {}).get("dim"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                message="商品名称集合 dense_vector 缺少合法维度",
                node_name="item_name_repository",
                cause=exc,
            ) from exc
        if actual_dim != self.config.embedding_dim:
            self._incompatible(
                "商品名称集合 dense_vector 维度不兼容: "
                f"expected={self.config.embedding_dim}, actual={actual_dim}"
            )

    def _validate_indexes(self, client: MilvusClient, collection_name: str) -> None:
        """校验两个检索索引及其度量方式。

        Collection 字段正确并不代表检索配置正确；如果 sparse 索引误用了其他 metric，
        数据仍可能写入，但查询分数语义会变化，因此启动写入前必须一并拒绝。
        """
        required = {
            "dense_vector_index": ("dense_vector", "COSINE"),
            "sparse_vector_index": ("sparse_vector", "IP"),
        }
        try:
            index_names = set(client.list_indexes(collection_name=collection_name))
            for index_name, (expected_field, expected_metric) in required.items():
                if index_name not in index_names:
                    self._incompatible(f"商品名称集合缺少索引: {index_name}")
                description = client.describe_index(
                    collection_name=collection_name,
                    index_name=index_name,
                )
                metric = description.get("metric_type")
                if metric is None and isinstance(description.get("params"), dict):
                    metric = description["params"].get("metric_type")
                if str(metric).upper() != expected_metric:
                    self._incompatible(f"商品名称集合索引度量不兼容: {index_name}")
                actual_field = description.get("field_name")
                if actual_field is not None and actual_field != expected_field:
                    self._incompatible(f"商品名称集合索引字段不兼容: {index_name}")
        except ConfigurationError:
            raise
        except Exception as exc:
            raise MilvusError(
                message="校验 Milvus 商品名称集合索引失败",
                node_name="item_name_repository",
                cause=exc,
            ) from exc

    @staticmethod
    def _incompatible(message: str) -> None:
        raise ConfigurationError(message=message, node_name="item_name_repository")
