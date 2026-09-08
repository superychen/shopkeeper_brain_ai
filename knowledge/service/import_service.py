"""前端导入业务入口：校验文件 → 保存文件 → 提交后台任务；查询时读取状态。

API 层仅调用 upload/get_status，不参与文件或队列处理。
此模块只接收普通文件流和字符串，不依赖 FastAPI、Request 或 HTTPException。
后台真正执行 LangGraph 的逻辑继续由 ImportTaskFacade 负责。
"""

import asyncio
import hashlib
import logging
import re
import shutil
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from knowledge.schema.import_models import ImportSettings, UploadParameters
from knowledge.processor.import_processor.config import get_config
from knowledge.service.import_task_facade import ImportTaskFacade

logger = logging.getLogger("import.service")


class ImportRequestError(Exception):
    """业务错误仅携带分类和说明；由 API 层决定对应的 HTTP 状态码。"""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ImportService:
    """协调上传、查询及临时文件清理，不重复实现已有图节点。"""

    def __init__(self, settings: ImportSettings, facade=None):
        """接收已通过 Pydantic 校验的配置；facade 可替换为测试用任务门面。"""
        self.runtime = settings.runtime_dir.resolve()
        self.limit = settings.max_file_bytes
        self.max_disk_bytes = settings.max_disk_bytes
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.tasks = facade or ImportTaskFacade(
            get_config(), capacity=settings.queue_capacity, retention_seconds=settings.retention_seconds)
        self.retention = settings.retention_seconds
        self.stop_cleanup = asyncio.Event()
        self.cleaner = None

    def start(self):
        """由应用启动事件调用一次，开启清理协程。"""
        self.cleaner = asyncio.create_task(self._maintenance())

    async def close(self):
        """先停止清理，再关闭任务门面；关闭等待不占用 HTTP 事件循环。"""
        self.stop_cleanup.set()
        try:
            if self.cleaner:
                await self.cleaner
        finally:
            await asyncio.to_thread(self.tasks.close)

    async def _maintenance(self):
        """每分钟清理过期终态目录；文件系统故障只记录日志，下轮继续。"""
        while not self.stop_cleanup.is_set():
            try:
                await asyncio.to_thread(clean_expired, self.runtime, self.tasks.store, self.retention)
            except OSError:
                logger.warning("临时目录清理暂时失败")
            try:
                await asyncio.wait_for(self.stop_cleanup.wait(), timeout=60)
            except TimeoutError:
                pass

    def upload(self, filename, source, document_id=""):
        """接收普通二进制文件流并返回接纳结果，不等待模型处理完成。

        流的生命周期由调用方管理；本方法返回前完成落盘，后台只使用新文件路径。
        同步方法由 FastAPI 的 def 路由在线程池执行，因此文件读写不会阻塞轮询。
        """
        # 【流程 03 · 接纳上传】顺序为 Pydantic 校验 → 预留名额 → 检查磁盘 → save_upload 落盘。
        # 随后用传入的 document_id 或文件摘要作为文档标识，交给 tasks.submit（流程 04）。
        started = time.monotonic()
        # 请求类型、长度和名称规则集中在模型中；此处只把校验错误转为既有业务错误。
        try:
            params = UploadParameters(filename=filename, document_id=document_id)
        except ValidationError as exc:
            error = exc.errors(include_input=False, include_context=False)[0]
            code = "unsupported_file" if error["type"] == "unsupported_file" else "invalid_input"
            field = ".".join(str(part) for part in error["loc"])
            raise ImportRequestError(code, f"{field}: {error['msg']}") from exc
        filename = params.filename
        document_id = params.document_id
        if not self.tasks.reserve():
            raise ImportRequestError("queue_full", "任务队列已满或正在关闭，请稍后重试")

        task_id = uuid.uuid4().hex
        task_dir = self.runtime / task_id
        path = task_dir / "original" / filename
        accepted = False
        try:
            # 先校验资源，再落盘；任何失败都在 finally 中释放预留的队列名额。
            if not self._disk_available():
                raise ImportRequestError("unavailable", "导入临时目录空间不足，请稍后重试")
            checksum = save_upload(source, path, self.limit)
            document_id = document_id or checksum
            self.tasks.submit(task_id, document_id, path, time.monotonic() - started)
            accepted = True
            logger.info("上传任务已接纳: task_id=%s", task_id)
            return dict(task_id=task_id, document_id=document_id, status="queued",
                        status_url=f"/status/{task_id}", poll_interval_ms=1500,
                        total_steps=10 if path.suffix.lower() == ".pdf" else 9)
        except ImportRequestError:
            raise
        except Exception as exc:
            logger.error("上传接纳失败: task_id=%s, error_type=%s", task_id, type(exc).__name__)
            raise ImportRequestError("unavailable", "暂时无法接纳导入任务") from exc
        finally:
            if not accepted:
                self.tasks.slots.release()
                # UUID 目录只由服务端生成，清理前再验证父目录，不能删除客户端路径。
                if task_dir.parent == self.runtime and task_dir.exists():
                    try:
                        shutil.rmtree(task_dir)
                    except OSError:
                        logger.warning("未接纳文件清理失败: task_id=%s", task_id)

    def get_status(self, task_id):
        """返回状态快照；这里不查询数据库，也不等待模型执行锁。"""
        # 【流程 10 · 查询服务】从 tasks.store.snapshot 获取独立副本，再交回接口序列化。
        snapshot = self.tasks.store.snapshot(task_id)
        if snapshot is None:
            raise ImportRequestError("not_found", "任务不存在、已过期或服务已重启")
        return snapshot

    def _disk_available(self):
        """检查临时目录接纳阈值，不把它当作 PDF 解包后的硬磁盘配额。"""
        used = 0
        for path in self.runtime.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    used += path.stat().st_size
            except FileNotFoundError:
                # 其他任务可能刚完成清理，不影响本次检查。
                continue
        return used + self.limit <= self.max_disk_bytes and shutil.disk_usage(self.runtime).free > self.limit


def save_upload(source, path, limit):
    """在工作线程分块落盘、计算摘要及校验内容，完成后原子改名。

    source 是请求期间仍打开的临时文件；此函数结束后后台任务只使用 path。
    Markdown 用增量解码器校验 UTF-8，避免一次读入大文档。
    """
    # 【流程 03.1 · 保存文件】边读边校验大小、内容和摘要，完整保存后才把 .part 改成正式文件。
    # 返回摘要给 upload；后台不持有请求文件流，避免 HTTP 请求结束后文件被框架关闭。
    import codecs
    decoder = codecs.getincrementaldecoder("utf-8")()
    is_pdf = path.suffix.lower() == ".pdf"
    digest = hashlib.sha256()
    size = 0
    path.parent.mkdir(parents=True)
    partial = path.with_suffix(path.suffix + ".part")
    with partial.open("wb") as target:
        while block := source.read(1024 * 1024):
            if size == 0 and is_pdf and not block.startswith(b"%PDF-"):
                raise ImportRequestError("invalid_input", "文件内容不是有效的 PDF")
            size += len(block)
            if size > limit:
                raise ImportRequestError("file_too_large", "文件超过大小限制")
            if not is_pdf:
                try:
                    decoder.decode(block)
                except UnicodeDecodeError:
                    raise ImportRequestError("invalid_input", "Markdown 必须使用 UTF-8 编码") from None
            digest.update(block)
            target.write(block)
    if not size:
        raise ImportRequestError("invalid_input", "不能上传空文件")
    if not is_pdf:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raise ImportRequestError("invalid_input", "Markdown 编码不完整") from None
    partial.replace(path)
    return digest.hexdigest()


def clean_expired(runtime, store, retention):
    """清理超过保留期的 UUID 任务目录，运行中或仍排队的任务一律跳过。

    仅扫描服务自己的 runtime 直接子目录，不跟随链接，也不读取 MinIO 归档。
    启动后遗留且没有内存记录的旧任务同样按保留期清理。
    """
    for folder in runtime.iterdir():
        if not re.fullmatch(r"[a-f0-9]{32}", folder.name) or folder.is_symlink() or not folder.is_dir():
            continue
        record = store.snapshot(folder.name)
        if record and record["status"] in {"queued", "processing"}:
            continue
        try:
            if time.time() - folder.stat().st_mtime > retention and folder.resolve().parent == runtime:
                shutil.rmtree(folder)
        except OSError:
            logger.warning("过期目录清理失败: task_id=%s", folder.name)


