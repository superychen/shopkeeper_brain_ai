"""集中创建和复用 AI 服务客户端。"""

from __future__ import annotations

import logging
import threading
from typing import ClassVar

from openai import OpenAI

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import ConfigurationError, LLMError
from knowledge.utils.client.base import BaseClientManager


logger = logging.getLogger("import.ai_clients")


class AIClients(BaseClientManager):
    """管理 VLM、LLM 及后续其他 AI 客户端的单例。"""

    name = "ai_clients"

    _vlm_client: ClassVar[OpenAI | None] = None
    _vlm_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_vlm(cls) -> OpenAI:
        """获取使用 DeepSeek OpenAI 兼容接口的 VLM 客户端。"""
        return cls._get_or_create(
            instance_name="_vlm_client",
            lock=cls._vlm_lock,
            factory=cls._create_vlm,
        )

    @classmethod
    def _create_vlm(cls) -> OpenAI:
        """读取 DeepSeek 配置并创建 VLM 客户端。"""
        config = get_config()
        api_base = cls._require_config(
            config=config,
            field_name="deepseek_api_base",
            env_name="DEEPSEEK_API_BASE",
        )
        api_key = cls._require_config(
            config=config,
            field_name="deepseek_api_key",
            env_name="DEEPSEEK_API_KEY",
        )

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=api_base.rstrip("/"),
            )
        except Exception as exc:
            logger.error(
                "创建 DeepSeek VLM 客户端失败: error_type=%s",
                type(exc).__name__,
            )
            raise LLMError(
                message="创建 DeepSeek VLM 客户端失败",
                node_name=cls.name,
                cause=exc,
            ) from exc

        logger.info(
            "DeepSeek VLM 客户端创建完成: api_base=%s",
            api_base.rstrip("/"),
        )
        return client

    @classmethod
    def _require_config(
        cls,
        config: ImportConfig,
        field_name: str,
        env_name: str,
    ) -> str:
        """读取必需配置，缺失时抛出不包含敏感值的配置异常。"""
        value = getattr(config, field_name, "")
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                message=f"缺少必需配置: {env_name}",
                node_name=cls.name,
            )
        return value.strip()
