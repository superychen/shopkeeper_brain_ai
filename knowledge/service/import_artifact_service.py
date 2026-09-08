"""文件归档服务：只负责 MinIO 对象与归档清单，不负责运行 LangGraph。

原件先备份，图运行后备份原始/处理版 Markdown 和图片，最后写清单。
对象路径由服务端生成，清单是连接原件、切片和转换产物的索引。
"""

import hashlib
import io
import json
import mimetypes
import logging
from pathlib import Path

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger("import.artifacts")


class ImportArtifactService:
    """复用现有 MinIO 连接；每次调用返回经过大小及摘要元数据核验的对象描述。"""

    def __init__(self, config):
        """保存现有 MinIO 配置，首次真正上传时才创建存储连接。"""
        self.config = config

    def put(self, path, key, role):
        """流式计算 SHA-256，再上传文件；不把大型 PDF 全部读入内存。"""
        # 【流程 05 / 08 共用】上传一个对象并核验大小和摘要元数据，返回对象描述供清单记录。
        path = Path(path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        size = path.stat().st_size
        checksum = digest.hexdigest()
        client = StorageClients.get_minio(self.config)
        client.fput_object(self.config.minio_bucket, key, str(path),
                           content_type="text/markdown; charset=utf-8" if role in {"raw", "processed"} else
                           mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                           metadata={"sha256": checksum})
        self._verify(client, key, size, checksum)
        logger.info("文件归档完成: role=%s, bytes=%d", role, size)
        return dict(role=role, bucket=self.config.minio_bucket, object_key=key,
                    size=size, sha256=checksum)

    def _verify(self, client, key, size, checksum):
        """ETag 不等于 SHA-256，核对上传的摘要元数据与实际对象大小。"""
        info = client.stat_object(self.config.minio_bucket, key)
        metadata = {k.lower(): v for k, v in info.metadata.items()}
        if info.size != size or metadata.get("x-amz-meta-sha256") != checksum:
            raise ValueError("MinIO 归档核验失败")

    def finish(self, context, state, objects):
        """归档转换产物并最后提交清单，必需文件缺失时直接失败。"""
        # 【流程 08 · 产物归档】按原始 Markdown → 处理版 Markdown → 图片 → manifest 的顺序上传。
        # 所有必需对象核验通过才返回门面，由流程 09 发布 completed。
        for field, role in (("raw_md_path", "raw"), ("md_path", "processed")):
            objects.append(self.put(state[field], context[role], role))
        image_dir = Path(state["raw_md_path"]).parent / "images"
        if image_dir.exists():
            for path in sorted(image_dir.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    relative = path.relative_to(image_dir).as_posix()
                    objects.append(self.put(path, context["prefix"] + "/converted/images/" + relative, "image"))
        # 普通字典就是清单的建造过程，不引入链式 Builder 或额外继承层。
        manifest = dict(document_id=context["document_id"], task_id=context["task_id"],
                        filename=context["filename"], objects=objects,
                        written_chunk_count=state["written_chunk_count"],
                        chunks_collection=state["chunks_collection"],
                        embedding_model=state["embedding_model"],
                        warnings=state.get("import_warnings", []))
        self._write_manifest(context, manifest)
        return dict(bucket=self.config.minio_bucket, object_key=context["manifest"], objects=objects)

    def record_failure(self, context, objects, step, work):
        """失败后尽量保存已经生成的原始 Markdown 与资源，不运行剩余图节点。"""
        # 【异常补偿】门面已发布 failed 后才调用这里，尽量保留工作目录产物与失败清单。
        # 补偿报错只由门面记录警告，不得覆盖首次失败原因或把任务重新改成 processing。
        # PDF 原始转换文件即使没有走到图片节点，也应能从工作目录找回。
        for path in sorted(work.rglob("*")):
            if path.is_file() and not path.is_symlink():
                key = context["prefix"] + "/partial/" + path.relative_to(work).as_posix()
                objects.append(self.put(path, key, "partial"))
        self._write_manifest(context, dict(task_id=context["task_id"], document_id=context["document_id"],
                                           status="failed", failed_step=step, objects=objects))

    def _write_manifest(self, context, manifest):
        """最后上传小型 JSON 清单；成功/失败路径共用完全一致的核验规则。"""
        payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
        checksum = hashlib.sha256(payload).hexdigest()
        client = StorageClients.get_minio(self.config)
        client.put_object(self.config.minio_bucket, context["manifest"], io.BytesIO(payload),
                          len(payload), content_type="application/json", metadata={"sha256": checksum})
        self._verify(client, context["manifest"], len(payload), checksum)
