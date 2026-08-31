"""集中创建和复用对象存储客户端。"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar
from urllib.parse import urlsplit

from minio import Minio

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    MinioError,
)
from knowledge.utils.client.base import BaseClientManager


logger = logging.getLogger("import.storage_clients")


class StorageClients(BaseClientManager):
    """管理 MinIO 及后续其他存储客户端的单例。"""

    name = "storage_clients"

    _minio_client: ClassVar[Minio | None] = None
    _minio_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_minio(cls) -> Minio:
        """获取 MinIO 客户端，并保证配置的 bucket 已经存在。"""
        return cls._get_or_create(
            instance_name="_minio_client",
            lock=cls._minio_lock,
            factory=cls._create_minio,
        )

    @classmethod
    def _create_minio(cls) -> Minio:
        """读取 MinIO 配置，创建客户端并按需初始化 bucket。"""
        config = get_config()
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
