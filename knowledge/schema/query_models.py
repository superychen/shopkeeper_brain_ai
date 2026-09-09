"""查询输入、提取证据和检索结果的运行时契约。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class QueryModel(BaseModel):
    """统一运行时校验：拒绝额外字段、去首尾空白、拒绝不符合严格类型的值。

    例如 original_query=123 不会被自动当作 "123"；格式错误应尽早暴露。
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class HistoryMessage(QueryModel):
    """一条由调用方提供的历史消息，不能带 system 角色来覆盖固定提取规则。"""
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class QueryInput(QueryModel):
    """【02】进入节点前校验的输入模型，与包含输出字段的图 state 区分。

    示例：QueryInput(original_query="RS-12 怎么用？")，history 默认独立空列表。
    单条/条数限制在这里，历史总字符预算由 QueryBaseNode._request 进一步检查。
    """
    original_query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="", max_length=128)
    request_id: str = Field(default="", max_length=128, pattern=r"^[a-zA-Z0-9_-]*$")
    history: list[HistoryMessage] = Field(default_factory=list, max_length=10)


class QueryMention(QueryModel):
    """【04】一次有原文证据的商品表述，不是已确认的库内产品。

    示例：name="RS-12"，evidence="RS-12"，source="current_query"。
    用户简称是否对应“RS PRO RS-12 数字万用表”，需要后续向量检索判断。
    """
    name: str = Field(min_length=1, max_length=256)
    evidence: str = Field(min_length=1, max_length=500)
    source: Literal["current_query", "history"]


class QueryItemExtraction(QueryModel):
    """LLM 的结构化输出：表述列表、改写草稿和是否需要拆分过多目标。

    rewritten_query 仅为提取草稿；最终对外改写由 aggregate 绑定库内名称生成。
    too_many_products=True 时直接请求拆分问题，不以截断后的 mentions 继续查询。
    """
    mentions: list[QueryMention] = Field(max_length=5)
    rewritten_query: str = Field(min_length=1, max_length=2000)
    too_many_products: bool = False


class ItemHit(QueryModel):
    """【06】一条文档级命中；score 已是融合结果，范围 0～1，不是向量某个分量。

    pk/document_id 标识来源；同商品多份文档可以有不同 pk，不能直接当成不同商品。
    """
    pk: str = Field(min_length=1, max_length=64)
    document_id: str = Field(min_length=1, max_length=128)
    item_name: str = Field(min_length=1, max_length=1024)
    file_title: str = Field(max_length=1024)
    embedding_model: str = Field(min_length=1, max_length=128)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class ItemSearchResult(QueryModel):
    """一次表述检索的所有命中及完整性标志。

    hits=[] 可以表示合法无匹配；数据库故障应抛 QueryError，而不是伪装成空 hits。
    truncated=True 表示仍可能有未召回的竞争候选，评分层应返回澄清。
    """
    hits: list[ItemHit]
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
