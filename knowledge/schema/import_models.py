"""导入请求与配置的 Pydantic 模型：只校验数据，不读取文件或调用外部服务。

阅读入口：UploadParameters 校验上传参数；ImportSettings 校验启动配置。
Field 表达长度和数值范围，field_validator 只补充字节长度、扩展名等特殊规则。
"""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator
from pydantic_core import PydanticCustomError


class UploadParameters(BaseModel):
    """服务执行前先构造本模型；非法参数不会占用队列或生成本地文件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Pydantic 自动处理类型、非空、长度与非法路径字符，服务无需再写相同的 if。
    filename: str = Field(min_length=1, max_length=160, pattern=r'^[^<>:"/\\|?*\x00-\x1f]+$')
    document_id: str = Field(default="", max_length=128, description="可选业务文档 ID，UTF-8 最多 128 字节")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, name: str) -> str:
        """保留中文文件名；排除 Windows 保留名，并限定支持的文档类型。"""
        reserved = {"CON", "PRN", "AUX", "NUL"}
        reserved.update(f"COM{i}" for i in range(10))
        reserved.update(f"LPT{i}" for i in range(10))
        if name.endswith((" ", ".")) or name.split(".")[0].upper() in reserved:
            raise ValueError("文件名非法，请使用普通文件名")
        if Path(name).suffix.lower() not in {".pdf", ".md", ".markdown"}:
            # 保留既有接口的 415 语义，API/服务根据错误分类进行转换。
            raise PydanticCustomError("unsupported_file", "仅支持 PDF 或 Markdown 文件")
        return name

    @field_validator("document_id", mode="before")
    @classmethod
    def strip_document_id(cls, value):
        """先去掉两端空白再执行长度校验；非字符串交给 Pydantic 报类型错误。"""
        return value.strip() if isinstance(value, str) else value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str) -> str:
        """Milvus 限制的是 UTF-8 字节，Field.max_length 限制的则是字符数。"""
        if "\0" in value or len(value.encode("utf-8")) > 128:
            raise ValueError("document_id 不能含 NUL，且 UTF-8 长度不能超过 128 字节")
        return value


class ImportSettings(BaseModel):
    """启动时一次校验导入配置，错误环境变量会直接阻止服务启动。

    default_factory 在构造时读取环境变量；validate_default=True 让这些字符串
    默认值也经过 Pydantic 转换和正整数校验，不需要手动 int(...) 和 <= 0 判断。
    """

    model_config = ConfigDict(validate_default=True, extra="forbid", frozen=True)

    runtime_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "runtime/imports")
    max_file_bytes: PositiveInt = Field(default_factory=lambda: os.getenv("IMPORT_MAX_FILE_BYTES", "104857600"))
    queue_capacity: PositiveInt = Field(default_factory=lambda: os.getenv("IMPORT_QUEUE_CAPACITY", "10"))
    retention_seconds: PositiveInt = Field(default_factory=lambda: os.getenv("IMPORT_RETENTION_SECONDS", "86400"))
    max_disk_bytes: PositiveInt = Field(default_factory=lambda: os.getenv("IMPORT_MAX_DISK_BYTES", "5368709120"))

    @field_validator("max_file_bytes", "queue_capacity", "retention_seconds", "max_disk_bytes", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        """环境变量字符串允许转整数，但显式传入 True 不应被当作容量 1。"""
        if isinstance(value, bool):
            raise ValueError("容量与保留时间必须为正整数，不能是布尔值")
        return value
