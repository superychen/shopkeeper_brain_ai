"""集中创建和复用 AI 服务客户端。

DeepSeek 文本/视觉模型通过远程 API 调用；BGE-M3 则在当前 Python 进程加载本地模型。
管理器只负责客户端生命周期和配置，商品名称抽取规则仍留在业务节点与 Prompt 中。

调用关系：图片节点 → get_vlm；商品名节点 → get_llm；两种编码服务 → get_bge_m3。
get_* 通过父类的锁与缓存复用实例，首次访问才执行 _create_*；实际推理由业务层调用。
同一进程共享第一次创建的客户端，后续传入不同配置不会自动重建已有实例。
"""

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar

from langchain_deepseek import ChatDeepSeek
from openai import OpenAI

from knowledge.processor.import_processor.config import ImportConfig, get_config
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    EmbeddingError,
    LLMError,
)
from knowledge.utils.client.base import BaseClientManager


logger = logging.getLogger("import.ai_clients")


class AIClients(BaseClientManager):
    """管理 VLM、LLM 及后续其他 AI 客户端的单例。"""

    name = "ai_clients"

    _vlm_client: ClassVar[OpenAI | None] = None
    _vlm_lock: ClassVar[threading.Lock] = threading.Lock()
    _llm_client: ClassVar[ChatDeepSeek | None] = None
    _llm_lock: ClassVar[threading.Lock] = threading.Lock()
    _bge_m3_client: ClassVar[Any | None] = None
    _bge_m3_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_vlm(cls, config: ImportConfig | None = None) -> OpenAI:
        """获取使用 DeepSeek OpenAI 兼容接口的 VLM 客户端。"""
        return cls._get_or_create(
            instance_name="_vlm_client",
            lock=cls._vlm_lock,
            factory=lambda: cls._create_vlm(config),
        )

    @classmethod
    def _create_vlm(cls, config: ImportConfig | None = None) -> OpenAI:
        """读取 DeepSeek 配置并创建视觉模型使用的原生客户端。"""
        config = config or get_config()
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
    def get_llm(cls, config: ImportConfig | None = None) -> ChatDeepSeek:
        """获取 LangChain DeepSeek 文本模型单例。

        单例避免每份文档重新创建 HTTP 连接池；实际请求仍在 LangChain ``invoke`` 时发生。
        """
        active_config = config or get_config()
        return cls._get_or_create(
            instance_name="_llm_client",
            lock=cls._llm_lock,
            factory=lambda: cls._create_llm(active_config),
        )

    @classmethod
    def _create_llm(cls, config: ImportConfig) -> ChatDeepSeek:
        """创建带超时和内置重试能力的 LangChain DeepSeek 模型。

        例如 DEEPSEEK_MAX_RETRIES=2 表示首次瞬时失败后最多再尝试两次；结构化输出契约
        由调用方通过 ``with_structured_output(ItemNameExtraction)`` 绑定。
        """
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
        model = cls._require_config(
            config=config,
            field_name="deepseek_llm_model",
            env_name="DEEPSEEK_LLM_MODEL",
        )

        try:
            # 商品名提取只需简短结构化结果，固定温度并关闭思考输出以控制响应形态。
            client = ChatDeepSeek(
                model=model,
                api_key=api_key,
                base_url=api_base.rstrip("/"),
                temperature=0.0,
                max_tokens=300,
                timeout=float(config.deepseek_timeout_seconds),
                max_retries=config.deepseek_max_retries,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as exc:
            logger.error(
                "创建 LangChain DeepSeek LLM 失败: model=%s, error_type=%s",
                model,
                type(exc).__name__,
            )
            raise LLMError(
                message="创建 LangChain DeepSeek LLM 失败",
                node_name=cls.name,
                cause=exc,
            ) from exc

        logger.info(
            "LangChain DeepSeek LLM 创建完成: model=%s, api_base=%s",
            model,
            api_base.rstrip("/"),
        )
        return client

    @classmethod
    def get_bge_m3(cls, config: ImportConfig | None = None) -> Any:
        """获取 BGE-M3 混合嵌入模型单例，避免每个文档重复加载大模型权重。"""
        active_config = config or get_config()
        return cls._get_or_create(
            instance_name="_bge_m3_client",
            lock=cls._bge_m3_lock,
            factory=lambda: cls._create_bge_m3(active_config),
        )

    @classmethod
    def _create_bge_m3(cls, config: ImportConfig) -> Any:
        """按实际设备创建 BGE-M3；CPU 场景强制关闭 FP16。

        ``auto`` 在有 CUDA 时选择 cuda:0，否则回退 CPU；FP16 只在 CUDA 上启用，
        避免 CPU 的 LayerNorm 等算子因半精度支持不足而运行失败。
        """
        model_name = cls._require_config(
            config=config,
            field_name="bge_m3_model_name",
            env_name="BGE_M3_MODEL_NAME",
        )
        configured_device = cls._require_config(
            config=config,
            field_name="bge_m3_device",
            env_name="BGE_M3_DEVICE",
        )

        try:
            import torch
            from pymilvus.model.hybrid import BGEM3EmbeddingFunction

            # 第一步：auto 根据硬件选择设备；显式指定的设备则交给模型加载器处理。
            device = configured_device
            if device.casefold() == "auto":
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            use_fp16 = device.casefold().startswith("cuda")
            # 第二步：只生成入库所需 dense/sparse；关闭未使用的 ColBERT 多向量输出。
            client = BGEM3EmbeddingFunction(
                model_name=model_name,
                device=device,
                use_fp16=use_fp16,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
        except Exception as exc:
            logger.error(
                "创建 BGE-M3 客户端失败: model=%s, device=%s, error_type=%s",
                model_name,
                configured_device,
                type(exc).__name__,
            )
            raise EmbeddingError(
                message="创建 BGE-M3 客户端失败",
                node_name=cls.name,
                cause=exc,
            ) from exc

        logger.info(
            "BGE-M3 客户端创建完成: model=%s, device=%s, use_fp16=%s",
            model_name,
            device,
            use_fp16,
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
