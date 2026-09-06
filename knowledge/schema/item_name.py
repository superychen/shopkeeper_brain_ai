"""商品名称结构化提取的数据模型。

示例：DeepSeek 返回 ``{"status": "recognized", "item_name":
"RS PRO RS-12 数字万用表", ...}`` 后，LangChain 会直接构造本模型；字段缺失、
置信度超过 1 或混入额外字段时，在业务节点执行前就会得到明确的校验错误。
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ItemNameExtraction(BaseModel):
    """DeepSeek 必须返回的商品名称提取结构和字段间约束。"""

    # extra="forbid" 可尽早发现模型擅自增加 explanation 等字段；frozen=True 则保证
    # 结构化结果在后续证据校验过程中不会被意外修改。
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    status: Literal["recognized", "not_found", "ambiguous"]
    item_name: str = Field(max_length=256)
    brand: str = Field(max_length=128)
    model: str = Field(max_length=128)
    product_type: str = Field(max_length=128)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(max_length=3)

    @field_validator("item_name", "brand", "model", "product_type")
    @classmethod
    def validate_single_line(cls, value: str) -> str:
        """名称字段必须保持单行，避免解释文字混入结构化结果。"""
        if "\n" in value or "\r" in value:
            raise ValueError("商品名称字段不能包含换行")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, evidence: list[str]) -> list[str]:
        """证据必须是短小、非空的原文片段。"""
        if any(not item.strip() or len(item) > 100 for item in evidence):
            raise ValueError("evidence 每项必须是长度不超过 100 的非空字符串")
        return evidence

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ItemNameExtraction":
        """识别状态决定名称和证据字段是否允许出现。

        例如 ``not_found`` 必须配合空 item_name 和空 evidence，防止上游把“未找到”
        当成可入库商品；``recognized`` 则至少要有名称和一条可回查的原文证据。
        """
        name_fields = (self.item_name, self.brand, self.model, self.product_type)
        if self.status == "recognized":
            if not self.item_name or not self.evidence:
                raise ValueError("recognized 状态必须包含 item_name 和 evidence")
        elif any(name_fields) or self.evidence:
            raise ValueError("未识别或歧义状态不能携带商品名称和证据")
        return self
