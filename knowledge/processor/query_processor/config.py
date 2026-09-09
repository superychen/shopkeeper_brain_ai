"""查询业务配置与已有连接配置组合，避免触发导入专用配置校验。"""

from dataclasses import dataclass, field
import math
import os
import re
from typing import ClassVar

from knowledge.processor.import_processor.config import ImportConfig


def env_int(name: str, default: int) -> int:
    """读取环境变量整数；未配置使用默认值，非法文本直接报错而不偷偷回退。"""
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class QueryConfig:
    """查询参数在节点初始化时构造；shared 复用模型/连接/集合配置。

    frozen=True 禁止重新给本配置字段赋值，但不会冻结 shared 内部的 ImportConfig。
    default_factory 延迟到创建实例时读取环境变量；不同实例不共用同一个 shared 对象。
    """
    shared: ImportConfig = field(default_factory=ImportConfig)
    # 自动确认门槛属于本期业务契约，ClassVar 不会成为 dataclass 的构造参数。
    # 所以 QueryConfig(item_name_high_confidence=0.6) 不合法；该值也不读取环境变量。
    item_name_high_confidence: ClassVar[float] = 0.7
    item_name_mid_confidence: float = field(default_factory=lambda: float(os.getenv("QUERY_ITEM_MID_CONFIDENCE", "0.45")))
    item_name_score_gap: float = field(default_factory=lambda: float(os.getenv("QUERY_ITEM_SCORE_GAP", "0.15")))
    item_name_dense_weight: float = 0.5
    item_name_sparse_weight: float = 0.5
    # TopK 控制数据库召回记录数；max_options 控制最后展示多少个产品，两者不能混为一谈。
    item_name_top_k: int = field(default_factory=lambda: env_int("QUERY_ITEM_TOP_K", 50))
    item_name_max_top_k: int = field(default_factory=lambda: env_int("QUERY_ITEM_MAX_TOP_K", 200))
    item_name_max_options: int = 5
    item_name_max_extracted: int = 5
    max_query_chars: int = 2000
    max_history_messages: int = 10
    max_history_chars: int = 8000
    llm_max_tokens: int = field(default_factory=lambda: env_int("QUERY_LLM_MAX_TOKENS", 1200))

    def __post_init__(self):
        """dataclass 自动在初始化后调用，先拒绝非法配置，再开始任何外部请求。

        例：mid=0.7 会使候选区间失去意义；改变等权设置会改变 0.7 的分数含义，均拒绝。
        bool 是 Python int 的子类，因此数值参数要显式排除 True/False。
        """
        for value in (self.item_name_mid_confidence, self.item_name_score_gap,
                      self.item_name_dense_weight, self.item_name_sparse_weight):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("查询评分配置必须是有限数值")
        if not 0 <= self.item_name_mid_confidence < self.item_name_high_confidence:
            raise ValueError("候选阈值必须在 [0, 0.7) 内")
        if not 0 < self.item_name_score_gap <= 1:
            raise ValueError("候选分差必须在 (0, 1] 内")
        if (self.item_name_dense_weight, self.item_name_sparse_weight) != (0.5, 0.5):
            raise ValueError("本期融合权重固定为 0.5/0.5，改变权重需要重新验收评分契约")
        for name in ("item_name_top_k", "item_name_max_top_k", "item_name_max_options",
                     "item_name_max_extracted", "max_query_chars", "max_history_messages",
                     "max_history_chars", "llm_max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if not self.item_name_max_options <= self.item_name_top_k <= self.item_name_max_top_k <= 16384:
            raise ValueError("候选数、TopK 和最大 TopK 配置不兼容")
        if self.item_name_max_extracted > 5:
            raise ValueError("本期最多同时确认 5 个商品")
        timeout = self.shared.milvus_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Milvus 查询超时必须为有限正数")
        collection = self.shared.item_name_collection
        if (not isinstance(collection, str) or len(collection) > 255
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", collection)
                or collection == self.shared.chunks_collection):
            raise ValueError("商品名集合名称非法，或与切片集合重复")
