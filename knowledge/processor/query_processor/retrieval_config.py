"""三路召回独立配置：不改变商品确认的 0.7 门槛。

构建顺序：ImportConfig 加载已有环境 → QueryConfig → RetrievalConfig 校验召回参数。
例：RetrievalConfig(web_enabled=False) 只关闭网络，本地向量/HyDE照常运行；
RetrievalConfig(chunks_per_item=5) 表示每商品每路最多5条，而不是三路总共5条。
带 default_factory 的字段在实例创建时读取环境；其他默认参数通过构造函数覆盖。
"""
from dataclasses import dataclass, field
import math
import os
from pydantic import SecretStr

from knowledge.processor.query_processor.config import QueryConfig


def enabled(name):
    """严格读取 true/false 开关，避免把字符串 'false' 当成真值。

    例：环境 QUERY_WEB_ENABLED=False → False；未配置 → True；值为 yes 会报错。
    lower 只统一大小写，不清除额外空格，因此配置拼写错误不会被静默接受。
    """
    value = os.getenv(name, "true").lower()
    if value not in {"true", "false"}:
        raise ValueError(f"{name} 必须是 true/false")
    return value == "true"


@dataclass(frozen=True)
class RetrievalConfig:
    """本轮图的不可重新赋值配置；shared 仍是旧配置对象，不是深度冻结副本。

    query 复用确认/连接参数但不出现在 repr，防止其内部凭据被间接打印。
    api_key 使用 SecretStr 包装，仅建立认证头时解包；不能记录解包结果。
    示例：dense_top_k=50、sparse_top_k=50、chunks_per_item=5，表示各ANN先召回50，
    Milvus融合后最多保留5。权重顺序与 dense/sparse 请求顺序一致。
    """
    query: QueryConfig = field(default_factory=QueryConfig, repr=False)
    # 召回候选数与最终条数是不同阶段；后者必须不超过前两者。
    dense_top_k: int = field(default_factory=lambda: int(os.getenv("QUERY_DENSE_TOP_K", "50")))
    sparse_top_k: int = field(default_factory=lambda: int(os.getenv("QUERY_SPARSE_TOP_K", "50")))
    chunks_per_item: int = field(default_factory=lambda: int(os.getenv("QUERY_CHUNKS_PER_ITEM", "5")))
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    hyde_enabled: bool = field(default_factory=lambda: enabled("QUERY_HYDE_ENABLED"))
    web_enabled: bool = field(default_factory=lambda: enabled("QUERY_WEB_ENABLED"))
    web_results_per_item: int = 3
    # 单位为秒：整路预算覆盖排队/调用/汇总，单次调用预算限制一次外部等待。
    # HyDE 包含一次额外生成，所以比原问题路预算更大；这些不是性能保证。
    vector_timeout: float = 20
    hyde_timeout: float = 45
    web_timeout: float = 25
    call_timeout: float = 10
    llm_timeout: float = 15
    # 必须是目标服务外部调用地址而非控制台页面；这里允许空值，网络路再单独报错，
    # 从而不会因为网络未配置导致本地图无法构建。
    mcp_url: str = field(default_factory=lambda: os.getenv("BAILIAN_SEARCH_MCP_URL", ""))
    tool_name: str = field(default_factory=lambda: os.getenv("BAILIAN_SEARCH_TOOL_NAME", ""))
    api_key: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("DASHSCOPE_API_KEY", "")), repr=False)

    @property
    def shared(self):
        """沿用 cfg.shared 的访问方式：例如 cfg.shared.chunks_collection 是正文集合名。"""
        return self.query.shared

    def __post_init__(self):
        """dataclass 构造后自动运行，外呼前拒绝不一致配置。

        例：chunks_per_item=60 而 dense_top_k=50 会报错；timeout=NaN/0会报错。
        bool 是 int 的子类，所以数量用 type(value) is int，避免 True 被当成1。
        等权设置固定，改变融合比例需要重新评估检索效果，不在这里隐式归一化。
        """
        if type(self.hyde_enabled) is not bool or type(self.web_enabled) is not bool:
            raise ValueError("检索开关必须为布尔值")
        for name in ("dense_top_k", "sparse_top_k", "chunks_per_item", "web_results_per_item"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f"{name} 超出范围")
        if self.chunks_per_item > min(self.dense_top_k, self.sparse_top_k):
            raise ValueError("最终条数不能超过两路召回条数")
        for name in ("vector_timeout", "hyde_timeout", "web_timeout", "call_timeout", "llm_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or not 0 < value <= 60:
                raise ValueError(f"{name} 超出范围")
        if (self.dense_weight, self.sparse_weight) != (0.5, 0.5):
            raise ValueError("本期正文融合采用等权设置")
