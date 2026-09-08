"""上传与轮询的响应结构，只定义接口字段，不执行任何业务。"""

from typing import Literal
from pydantic import BaseModel


class UploadAccepted(BaseModel):
    """上传接纳响应；明确告诉前端去哪查询，而不是返回整个图状态。"""
    task_id: str
    document_id: str
    status: Literal["queued"]
    status_url: str
    poll_interval_ms: int
    total_steps: int


class StepStatus(BaseModel):
    """稳定的节点 ID 用于程序判断，label 用于中文展示。"""
    id: str
    label: str
    status: Literal["pending", "running", "completed", "failed", "skipped"]
    duration_seconds: float


class TaskStatusResponse(BaseModel):
    """轮询协议：响应模型同时生成 OpenAPI 文档并过滤内部字段。"""
    task_id: str
    document_id: str
    filename: str
    status: Literal["queued", "processing", "completed", "failed"]
    current_step: str | None
    steps: list[StepStatus]
    done_list: list[str]
    running_list: list[str]
    skipped_list: list[str]
    durations: dict[str, float]
    total_steps: int
    finished_steps: int
    progress_percent: int
    elapsed_seconds: float
    queue_wait_seconds: float
    warnings: list[str]
    result: dict | None
    error: dict | None


