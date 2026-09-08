"""保存前端轮询需要的轻量状态，不保存文档正文和向量。

后台线程更新记录，HTTP 线程读取快照；所有操作共用一把短锁。
锁内只操作字典，绝不执行模型、文件或网络调用。
"""

import copy
import threading
import time


STEPS = [
    ("upload_file", "接收上传文件"), ("backup_original", "备份原件并准备工作副本"),
    ("entry_node", "检查导入文件"), ("pdf_to_md_node", "PDF 转 Markdown"),
    ("md_to_img_node", "图片摘要与资源处理"), ("document_split_node", "文档切片"),
    ("item_name_recognition_node", "商品名称识别"),
    ("bge_embedding_chunks_node", "切片向量化"),
    ("milvus_import_node", "向量写入与核验"), ("backup_artifacts", "归档转换产物"),
]


class TaskStore:
    """一个进程共用一个实例；服务重启后记录消失，不承诺断点恢复。"""

    def __init__(self, retention_seconds=86400):
        """设置终态保留时间，运行中的记录不会因为超时被清理。"""
        if retention_seconds <= 0:
            raise ValueError("任务保留时间必须为正数")
        self.records = {}
        self.lock = threading.Lock()
        self.retention_seconds = retention_seconds

    def create(self, task_id, document_id, filename, upload_seconds):
        """接纳上传后建立任务，PDF 转换只对 PDF 文件适用。"""
        now = time.monotonic()
        steps = []
        for step_id, label in STEPS:
            status = "pending"
            if step_id == "upload_file":
                status = "completed"
            elif step_id == "pdf_to_md_node" and not filename.lower().endswith(".pdf"):
                status = "skipped"
            steps.append(dict(id=step_id, label=label, status=status,
                              duration_seconds=upload_seconds if step_id == "upload_file" else 0))
        with self.lock:
            self._expire()
            self.records[task_id] = dict(
                task_id=task_id, document_id=document_id, filename=filename, status="queued",
                steps=steps, warnings=[], result=None, error=None,
                created=now - upload_seconds, queued=now, started=None, ended=None,
            )

    def start(self, task_id):
        """工作线程实际取到任务时才进入 processing，排队时间单独统计。"""
        with self.lock:
            record = self.records[task_id]
            record.update(status="processing", started=time.monotonic())

    def step(self, task_id, step_id, status):
        """原子记录一次步骤事件；结束耗时固定，运行耗时由查询动态计算。"""
        with self.lock:
            record = self.records[task_id]
            if record["status"] in {"completed", "failed"}:
                return
            step = next(s for s in record["steps"] if s["id"] == step_id)
            now = time.monotonic()
            if status == "running":
                step["started"] = now
            else:
                step["duration_seconds"] = now - step.get("started", now)
            step["status"] = status

    def finish(self, task_id, *, result=None, warnings=None, error=None):
        """任务门面统一提交终态；任何失败都必须清空运行中的步骤。"""
        # 【流程 09 / 异常出口】一次性提交任务终态；失败时把还在 running 的步骤一起标为 failed。
        # 后续补偿不再修改这个终态，前端流程 10 读到 completed/failed 即停止轮询。
        with self.lock:
            record = self.records[task_id]
            if record["status"] in {"completed", "failed"}:
                return
            now = time.monotonic()
            for step in record["steps"]:
                if step["status"] == "running":
                    step.update(status="failed", duration_seconds=now - step["started"])
            record.update(status="failed" if error else "completed", ended=now,
                          result=result, warnings=warnings or [], error=error)

    def snapshot(self, task_id):
        """返回独立副本，调用方不能通过修改响应影响后台记录。"""
        # 【流程 10 · 读取快照】短锁内复制状态，锁外计算耗时和进度，再经 service → route 返回页面。
        with self.lock:
            self._expire()
            record = copy.deepcopy(self.records.get(task_id))
        if record is None:
            return None
        now = record["ended"] or time.monotonic()
        for step in record["steps"]:
            if step["status"] == "running":
                step["duration_seconds"] = now - step["started"]
            step.pop("started", None)
        steps = record["steps"]
        total = sum(s["status"] != "skipped" for s in steps)
        done = [s["id"] for s in steps if s["status"] == "completed"]
        running = [s["id"] for s in steps if s["status"] == "running"]
        record.update(done_list=done, running_list=running,
                      skipped_list=[s["id"] for s in steps if s["status"] == "skipped"],
                      current_step=running[0] if running else None,
                      durations={s["id"]: round(s["duration_seconds"], 3) for s in steps},
                      total_steps=total, finished_steps=len(done),
                      progress_percent=100 if record["status"] == "completed" else min(99, int(len(done) / total * 100)),
                      elapsed_seconds=round(now - record["created"], 3),
                      queue_wait_seconds=round((record["started"] or now) - record["queued"], 3))
        for key in ("created", "queued", "started", "ended"):
            record.pop(key)
        return record

    def _expire(self):
        """仅在持锁时调用，清理过期终态，保持运行任务可查询。"""
        now = time.monotonic()
        for task_id, record in list(self.records.items()):
            if record["ended"] and now - record["ended"] > self.retention_seconds:
                del self.records[task_id]
