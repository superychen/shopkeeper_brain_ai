"""集中创建和复用存储连接，不承载切片组装或入库核验等业务规则。

图片上传组件 → get_minio → 首次创建连接并确保 bucket 存在。
商品名/切片仓储 → get_milvus → 首次创建连接 → 仓储负责 schema 和数据操作。
父类使用锁缓存实例，进程内后续请求复用首次配置；修改配置不会自动重建连接。
"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar
from urllib.parse import urlsplit

from minio import Minio
from pymilvus import MilvusClient

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    MilvusError,
    MinioError,
)
from knowledge.utils.client.base import BaseClientManager


logger = logging.getLogger("import.storage_clients")


class StorageClients(BaseClientManager):
    """管理 MinIO 和 Milvus 客户端单例。"""

    name = "storage_clients"

    _minio_client: ClassVar[Minio | None] = None
    _minio_lock: ClassVar[threading.Lock] = threading.Lock()
    _milvus_client: ClassVar[MilvusClient | None] = None
    _milvus_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_minio(cls, config: ImportConfig | None = None) -> Minio:
        """获取 MinIO 客户端，并保证配置的 bucket 已经存在。"""
        return cls._get_or_create(
            instance_name="_minio_client",
            lock=cls._minio_lock,
            factory=lambda: cls._create_minio(config),
        )

    @classmethod
    def _create_minio(cls, config: ImportConfig | None = None) -> Minio:
        """读取 MinIO 配置，创建客户端并按需初始化 bucket。"""
        config = config or get_config()
        endpoint, secure = cls._parse_endpoint(config)
        access_key = cls._require_config(
            config=config,
            field_name="minio_access_key",
            env_name="MINIO_ACCESS_KEY",
        )
        secret_key = cls._require_config(
            config=config,
            field_name="minio_secret_key",
            env_name="MINIO_SECRET_KEY",
        )
        bucket = cls._require_config(
            config=config,
            field_name="minio_bucket",
            env_name="MINIO_BUCKET_NAME",
        )

        try:
            client = Minio(
                endpoint=endpoint,
                access_key=access_key,
                secret_key=secret_key,
                secure=secure,
            )
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                logger.info("MinIO bucket 创建完成: bucket=%s", bucket)
        except Exception as exc:
            logger.error(
                "创建或连接 MinIO 客户端失败: endpoint=%s, bucket=%s, "
                "error_type=%s",
                endpoint,
                bucket,
                type(exc).__name__,
            )
            raise MinioError(
                message=f"创建或连接 MinIO 客户端失败: {endpoint}",
                node_name=cls.name,
                cause=exc,
            ) from exc

        logger.info(
            "MinIO 客户端创建完成: endpoint=%s, bucket=%s, secure=%s",
            endpoint,
            bucket,
            secure,
        )
        return client

    @classmethod
    def get_milvus(cls, config: ImportConfig | None = None) -> MilvusClient:
        """获取 Milvus 客户端单例，复用底层连接而不是按文档重复建立连接。"""
        active_config = config or get_config()
        return cls._get_or_create(
            instance_name="_milvus_client",
            lock=cls._milvus_lock,
            factory=lambda: cls._create_milvus(active_config),
        )

    @classmethod
    def _create_milvus(cls, config: ImportConfig) -> MilvusClient:
        """创建 Milvus 客户端，日志中不暴露鉴权 token。

        例如本地 standalone 使用 ``http://127.0.0.1:19530`` 且 token 留空；连接远程
        Milvus 时可沿用同一代码，只需在 .env 替换 URL 并提供服务端要求的 token。
        """
        uri = cls._require_config(
            config=config,
            field_name="milvus_url",
            env_name="MILVUS_URL",
        )
        token = config.milvus_token.strip() if config.milvus_token else ""
        # 仅在配置非空时传 token，兼容本地无鉴权实例；kwargs 不得写入日志。
        client_kwargs = {"uri": uri}
        if token:
            client_kwargs["token"] = token

        try:
            # ** 将字典展开为关键字参数，相当于逐个传入 uri=...、token=...。
            client = MilvusClient(**client_kwargs)
        except Exception as exc:
            logger.error(
                "创建 Milvus 客户端失败: endpoint=%s, error_type=%s",
                cls._safe_endpoint(uri),
                type(exc).__name__,
            )
            raise MilvusError(
                message="创建 Milvus 客户端失败",
                node_name=cls.name,
                cause=exc,
            ) from exc

        logger.info("Milvus 客户端创建完成: endpoint=%s", cls._safe_endpoint(uri))
        return client

    @staticmethod
    def _safe_endpoint(uri: str) -> str:
        """日志省略路径、查询参数与片段，仅保留协议和 netloc。

        鉴权应通过独立 token 配置传入；本函数不会额外清洗 netloc 内的用户信息。
        """
        parsed = urlsplit(uri)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
        return "configured-local-endpoint"

    @classmethod
    def _parse_endpoint(cls, config: ImportConfig) -> tuple[str, bool]:
        """转换 endpoint；URL 中的协议优先于 MINIO_SECURE。"""
        raw_endpoint = cls._require_config(
            config=config,
            field_name="minio_endpoint",
            env_name="MINIO_ENDPOINT",
        )
        if "://" not in raw_endpoint:
            return raw_endpoint.rstrip("/"), config.minio_secure

        parsed = urlsplit(raw_endpoint)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc:
            raise ConfigurationError(
                message="MINIO_ENDPOINT 必须是 host:port 或有效的 HTTP(S) URL",
                node_name=cls.name,
            )
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ConfigurationError(
                message="MINIO_ENDPOINT 不能包含路径、查询参数或锚点",
                node_name=cls.name,
            )
        return parsed.netloc, parsed.scheme.casefold() == "https"

    @classmethod
    def _require_config(
        cls,
        config: ImportConfig,
        field_name: str,
        env_name: str,
    ) -> str:
        """读取必需配置，异常和日志中不包含访问密钥。"""
        value = getattr(config, field_name, "")
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                message=f"缺少必需配置: {env_name}",
                node_name=cls.name,
            )
        return value.strip()
