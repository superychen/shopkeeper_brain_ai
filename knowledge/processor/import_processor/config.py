"""
导入流程配置管理模块

集中管理所有配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional
import os
from dotenv import load_dotenv

# 固定读取 knowledge/.env，避免启动目录变化导致配置加载结果不同。
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _get_env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，避免非空字符串 ``false`` 被误判为真。"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() in {"1", "true", "yes", "on"}


def _get_env_int(name: str, default: int) -> int:
    """读取整数环境变量，具体取值范围由使用它的节点校验。

    例如环境变量不存在时 ``_get_env_int("EMBEDDING_DIM", 1024)`` 返回 1024；
    若配置为非数字字符串则在启动构造配置时立即失败，避免错误值流入模型层。
    """
    raw_value = os.getenv(name)
    return default if raw_value is None else int(raw_value)


def _get_env_float(name: str, default: float) -> float:
    """读取浮点环境变量，允许超时等配置使用 ``30`` 或 ``30.5``。"""
    raw_value = os.getenv(name)
    return default if raw_value is None else float(raw_value)


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000  # 切片最大长度
    img_content_length: int = 200  # 图片上下文最大长度
    min_content_length: int = 500  # 合并短内容的最小长度
    overlap_sentences: int = 1  # 句子级切分时的重叠句数
    item_name_chunk_k: int = 3  # 商品名识别时使用的切片数量
    item_name_chunk_size: int = 2500  # 商品名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== LLM 配置 ====================
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    vl_model: str = field(
        default_factory=lambda: os.getenv("VL_MODEL", "")
    )
    item_model: str = field(
        default_factory=lambda: os.getenv("ITEM_MODEL", "")
    )
    default_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "")
    )

    # ==================== DeepSeek 配置 ====================
    deepseek_api_base: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_BASE", "")
    )
    deepseek_api_key: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    deepseek_vlm_model: str = field(
        default_factory=lambda: os.getenv(
            "DEEPSEEK_VLM_MODEL",
            "deepseek-v4-flash-vision-exp",
        )
    )
    deepseek_llm_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_LLM_MODEL", "")
    )
    deepseek_timeout_seconds: float = field(
        default_factory=lambda: _get_env_float("DEEPSEEK_TIMEOUT_SECONDS", 30.0)
    )
    # 该值是“首次请求之后”的最大重试次数；2 表示总尝试次数最多为 3。
    deepseek_max_retries: int = field(
        default_factory=lambda: _get_env_int("DEEPSEEK_MAX_RETRIES", 2)
    )

    # ==================== Milvus 配置 ====================
    milvus_url: str = field(
        default_factory=lambda: os.getenv("MILVUS_URL", "")
    )
    milvus_token: str = field(
        default_factory=lambda: os.getenv("MILVUS_TOKEN", "")
    )
    chunks_collection: str = field(
        default_factory=lambda: os.getenv("CHUNKS_COLLECTION", "")
    )
    item_name_collection: str = field(
        default_factory=lambda: os.getenv("ITEM_NAME_COLLECTION", "kb_item_names_v1")
    )
    entity_name_collection: str = field(
        default_factory=lambda: os.getenv("ENTITY_NAME_COLLECTION", "")
    )


    # ==================== MinIO 配置 ====================
    minio_endpoint: str = field(
        default_factory=lambda: os.getenv("MINIO_ENDPOINT", "")
    )
    minio_access_key: str = field(
        default_factory=lambda: os.getenv("MINIO_ACCESS_KEY", "")
    )
    minio_secret_key: str = field(
        default_factory=lambda: os.getenv("MINIO_SECRET_KEY", "")
    )
    minio_bucket: str = field(
        default_factory=lambda: os.getenv("MINIO_BUCKET_NAME", "")
    )
    minio_secure: bool = field(
        default_factory=lambda: _get_env_bool("MINIO_SECURE", False)
    )

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: _get_env_int("EMBEDDING_DIM", 1024)
    )
    embedding_batch_size: int = 8
    bge_m3_model_name: str = field(
        default_factory=lambda: os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3")
    )
    bge_m3_device: str = field(
        default_factory=lambda: os.getenv("BGE_M3_DEVICE", "auto")
    )

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15  # 图片总结 API 速率限制

    @classmethod
    def from_env(cls) -> "ImportConfig":
        """从环境变量加载配置"""
        return cls()

    def get_minio_base_url(self) -> str:
        """返回可用于拼接对象地址的 MinIO HTTP(S) 基础地址。"""
        endpoint = self.minio_endpoint.strip().rstrip("/")
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        base_protocol = "https://" if self.minio_secure else "http://"
        return base_protocol + endpoint


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig.from_env()
    return _config
