"""导入任务门面：HTTP 只提交任务，此处统一协调后台执行与最终成功条件。

使用标准库 Queue 和一个工作线程，避免同时加载多个 BGE 推理任务。
流程：原件归档 → 工作副本 → LangGraph → 产物归档 → 标记完成。
没有跨 Milvus/MinIO 事务，失败时保留已经写入的数据及本地文件供排查。
"""

import hashlib
import logging
import os
import queue
import shutil
import threading
from pathlib import Path

from knowledge.service.task_store import TaskStore
from knowledge.service.import_artifact_service import ImportArtifactService

logger = logging.getLogger("import.tasks")


class ImportTaskFacade:
    """任务门面同时管理轻量队列，不再拆一层仅转发方法的执行器。"""

    def __init__(self, config, capacity=10, runner=None, artifacts=None, retention_seconds=None):
        """runner/artifacts 可注入替身，让接口测试不必加载模型或连接数据库。"""
        if capacity <= 0:
            raise ValueError("IMPORT_QUEUE_CAPACITY 必须为正整数")
        self.store = TaskStore(retention_seconds=retention_seconds if retention_seconds is not None
                               else int(os.getenv("IMPORT_RETENTION_SECONDS", "86400")))
        self.config = config
        self.queue = queue.Queue(maxsize=capacity)
        self.slots = threading.BoundedSemaphore(capacity)
        self.stopping = threading.Event()
        self.lifecycle_lock = threading.Lock()
        self.runner = runner
        self.artifacts = artifacts or ImportArtifactService(config)
        self.thread = threading.Thread(target=self._work, name="rag-import", daemon=True)
        self.thread.start()

    def reserve(self):
        """上传前预留容量，避免并发请求全部落盘后才发现队列已满。"""
        return not self.stopping.is_set() and self.slots.acquire(blocking=False)

    def submit(self, task_id, document_id, path, upload_seconds):
        """调用者已预留容量；登记状态后入队，后台仅持有路径，不持有 UploadFile。"""
        # 【流程 04 · 入队】先登记 queued 再入队，保证后台取到任务时已经有状态记录。
        # 提交后 upload 返回 task_id；_work 工作线程会独立取走此任务。
        with self.lifecycle_lock:
            if self.stopping.is_set():
                raise RuntimeError("导入服务正在关闭")
            self.store.create(task_id, document_id, path.name, upload_seconds)
            self.queue.put_nowait((task_id, document_id, path))

    def close(self):
        """停止接纳；等待中的任务标记失败，当前任务给 5 秒退出宽限。"""
        with self.lifecycle_lock:
            self.stopping.set()
            # 当前任务可能仍在模型中，直接在关闭线程终止排队状态，不等它跑完。
            while True:
                try:
                    job = self.queue.get_nowait()
                except queue.Empty:
                    break
                self.slots.release()
                self.store.finish(job[0], error=dict(code="SERVICE_STOPPED", message="服务关闭，请重新上传"))
                self.queue.task_done()
        self.thread.join(timeout=5)

    def _work(self):
        """只有此线程执行模型。单任务异常被隔离，下一任务仍能继续处理。"""
        # 【流程 04.1 · 后台入口】每次只取一个任务，调用 _run 完整执行后才取下一项。
        # 每个任务的异常在这里隔离，避免一个任务失败导致工作线程退出。
        while True:
            try:
                job = self.queue.get(timeout=0.2)
            except queue.Empty:
                if self.stopping.is_set():
                    return
                continue
            # 出队后释放等待名额；图内还有原有串行锁，CLI 也不会与其并行写入。
            self.slots.release()
            try:
                if self.stopping.is_set():
                    self.store.finish(job[0], error=dict(code="SERVICE_STOPPED", message="服务关闭，请重新上传"))
                else:
                    self._run(*job)
            except Exception as exc:
                logger.error("任务调度失败: task_id=%s, error_type=%s", job[0], type(exc).__name__)
                self.store.finish(job[0], error=dict(code="TASK_FAILED", message="后台任务失败"))
            finally:
                self.queue.task_done()

    def _run(self, task_id, document_id, path):
        """串联导入步骤，只有向量核验和归档均成功才提交 completed。"""
        self.store.start(task_id)
        key = hashlib.sha256(document_id.encode("utf-8")).hexdigest()
        prefix = f"imports/{key}/{task_id}"
        context = dict(document_id=document_id, task_id=task_id, filename=path.name,
                       prefix=prefix, bucket=self.config.minio_bucket,
                       original=f"{prefix}/original/{path.name}", raw=f"{prefix}/converted/raw.md",
                       processed=f"{prefix}/converted/processed.md", manifest=f"{prefix}/manifest.json")
        current = "backup_original"
        result = dict(vector_status="not_started", archive_status="pending")
        objects = []

        def observe(node, status):
            """图调用此简单回调，无需把 FastAPI 对象或锁放入图状态。"""
            # 【进度回调】main_graph.tracked 在每个节点执行前后调用这里；current 用于失败时定位步骤。
            # 这里只更新 TaskStore，HTTP 查询不会等待模型或存储调用。
            nonlocal current
            current = node
            if node == "milvus_import_node" and status == "running":
                result["vector_status"] = "unknown"
            self.store.step(task_id, node, status)

        try:
            # 【流程 05】原件备份和工作副本准备属于同一步，全部完成后才能报告成功。
            observe(current, "running")
            objects = [self.artifacts.put(path, context["original"], "original")]
            # 保留原文件不可变；MinerU 输出和 _new.md 都只能在 work 目录内产生。
            work = path.parent.parent / "work"
            work.mkdir(exist_ok=True)
            source = work / path.name
            shutil.copyfile(path, source)
            observe(current, "completed")
            logger.info("原件备份及工作副本准备完成: task_id=%s", task_id)
            # 【流程 06】把工作文件路径交给图；observe 将各节点进度写回同一份 TaskStore。
            if self.runner is None:
                from knowledge.processor.import_processor.main_graph import run_import_graph
                runner = run_import_graph
            else:
                runner = self.runner
            state = runner(dict(task_id=task_id, document_id=document_id, import_file_path=str(source)),
                           config=self.config, observer=observe, source_archive=context)
            result.update(vector_status="verified", written_chunk_count=state["written_chunk_count"],
                          chunks_collection=state["chunks_collection"], embedding_model=state["embedding_model"])
            # 【流程 08】图返回仅表示向量已核验；还要归档转换产物，任务才能成功。
            observe("backup_artifacts", "running")
            result["archive"] = self.artifacts.finish(context, state, objects)
            result["archive_status"] = "verified"
            observe("backup_artifacts", "completed")
            # 【流程 09】提交 completed 后，轮询即可结束；本地清理失败只记警告。
            self.store.finish(task_id, result=result, warnings=state.get("import_warnings", []))
            logger.info("导入任务完成: task_id=%s, chunks=%s", task_id, state["written_chunk_count"])
            # path 仅由上传接口构造；校验任务父目录名后才能递归清理。
            if path.parent.name == "original" and path.parent.parent.name == task_id:
                try:
                    shutil.rmtree(path.parent.parent)
                except OSError:
                    # 临时清理失败不改变已经核验的导入结果，定时清理会再次处理。
                    logger.warning("任务临时目录清理失败: task_id=%s", task_id)
        except Exception as exc:
            logger.error("导入任务失败: task_id=%s, step=%s, error_type=%s", task_id, current, type(exc).__name__)
            if result["archive_status"] != "verified":
                result["archive_status"] = "failed"
            # 保留期从本次失败开始，不能用长时间解析前的目录时间计算。
            try:
                os.utime(path.parent.parent, None)
            except OSError:
                logger.warning("无法更新失败目录保留时间: task_id=%s", task_id)
            # 【异常出口】先发布 failed 并结束 running 步骤，前端不必等待 MinIO 补偿。
            self.store.finish(task_id, result=result, error=dict(
                code="IMPORT_FAILED", step_id=current, message=f"步骤 {current} 执行失败，请检查服务日志", retryable=True))
            # 清单备份仍由当前工作线程尽力执行；失败不能覆盖已发布的原始错误。
            # 此时前端可停止轮询，但下一项排队任务仍需等这段补偿返回。
            if hasattr(self.artifacts, "record_failure"):
                try:
                    self.artifacts.record_failure(context, objects, current, path.parent.parent / "work")
                except Exception as archive_error:
                    logger.warning("失败任务清单备份未完成: task_id=%s, error_type=%s",
                                   task_id, type(archive_error).__name__)
