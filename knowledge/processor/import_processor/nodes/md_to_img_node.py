"""Markdown 本地图片处理节点的结构定义。"""

import logging
from dataclasses import dataclass
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import (
    FileProcessingError,
    StateFieldError,
)
from knowledge.processor.import_processor.state import ImportGraphState


@dataclass(frozen=True)
class _ImageContext:
    """保存一张图片在 Markdown 文档中的上下文。"""

    heading: str
    pre_text: str
    post_text: str


@dataclass(frozen=True)
class _ImageInfo:
    """保存图片文件及其 Markdown 上下文。"""

    name: str
    path: Path
    context: _ImageContext


class _MdFileHandler:
    """负责读取 Markdown、定位图片目录以及备份处理后的内容。"""

    def __init__(self, logger: logging.Logger, node_name: str) -> None:
        self.logger = logger
        self.node_name = node_name

    def read_md(self, state: ImportGraphState) -> tuple[str, Path, Path]:
        """读取 Markdown 内容，并返回内容、文件路径和图片目录。"""
        if not isinstance(state, dict):
            raise StateFieldError(
                node_name=self.node_name,
                field_name="state",
                expected_type=dict,
            )

        md_path_value = state.get("md_path")
        if not isinstance(md_path_value, str) or not md_path_value.strip():
            raise StateFieldError(
                node_name=self.node_name,
                field_name="md_path",
                expected_type=str,
            )

        md_path = Path(md_path_value.strip()).expanduser()
        if md_path.suffix.lower() not in {".md", ".markdown"}:
            raise StateFieldError(
                node_name=self.node_name,
                field_name="md_path",
                expected_type=str,
                message=f"状态字段 'md_path' 必须指向 Markdown 文件: {md_path}",
            )

        try:
            if not md_path.exists():
                raise FileProcessingError(
                    message=f"Markdown 文件不存在: {md_path}",
                    node_name=self.node_name,
                )
            if not md_path.is_file():
                raise FileProcessingError(
                    message=f"Markdown 路径不是普通文件: {md_path}",
                    node_name=self.node_name,
                )

            # 统一返回绝对路径，避免后续步骤因工作目录变化而定位到错误的图片目录。
            md_path = md_path.resolve()
            with md_path.open(mode="r", encoding="utf-8") as md_file:
                md_content = md_file.read()
        except FileProcessingError:
            raise
        except (OSError, UnicodeError) as exc:
            self.logger.error(
                "读取 Markdown 文件失败: md_path=%s, error=%s",
                md_path,
                exc,
            )
            raise FileProcessingError(
                message=f"读取 Markdown 文件失败: {md_path}",
                node_name=self.node_name,
                cause=exc,
            ) from exc

        image_dir = md_path.parent / "images"
        self.logger.info(
            "Markdown 文件读取完成: md_path=%s, content_length=%d, image_dir=%s",
            md_path,
            len(md_content),
            image_dir,
        )
        return md_content, md_path, image_dir

    def backup(self, md_path: Path, new_md_content: str) -> Path:
        """备份处理后的 Markdown 内容，并返回新文件路径。"""
        raise NotImplementedError("_MdFileHandler.backup 尚未实现")


class _ImageScanner:
    """负责扫描有效图片、定位引用并组装每张图片的上下文。"""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def scan_img_dir(
        self,
        image_dir: Path,
        md_content: str,
        image_extensions: set[str],
        context_length: int,
    ) -> list[_ImageInfo]:
        """扫描图片目录并返回包含 Markdown 上下文的图片集合。"""
        raise NotImplementedError("_ImageScanner.scan_img_dir 尚未实现")


class _VLMSummarizer:
    """负责根据图片及其上下文生成图片摘要。"""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def summarize_all(
        self,
        document_title: str,
        image_list: list[_ImageInfo],
        vl_model: str,
        requests_per_minute: int,
    ) -> dict[Path, str]:
        """为所有有效图片生成摘要。"""
        raise NotImplementedError("_VLMSummarizer.summarize_all 尚未实现")


class _ImageUploader:
    """负责上传图片并替换 Markdown 中的本地图片引用。"""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def upload_and_replace(
        self,
        document_name: str,
        md_content: str,
        image_summaries: dict[Path, str],
        image_list: list[_ImageInfo],
        minio_bucket: str,
        minio_base_url: str,
    ) -> str:
        """上传图片，并返回替换图片引用后的 Markdown 内容。"""
        raise NotImplementedError("_ImageUploader.upload_and_replace 尚未实现")


class MdToImgNode(BaseNode):
    """按既定顺序编排 Markdown 图片处理的四个协作类。"""

    name = "md_to_img_node"

    def __init__(self, config: ImportConfig | None = None) -> None:
        super().__init__(config=config)
        # 协作对象只在节点初始化时创建一次，process 仅负责串联处理步骤。
        self._file_handler = _MdFileHandler(self.logger, self.name)
        self._image_scanner = _ImageScanner(self.logger)
        self._vlm_summarizer = _VLMSummarizer(self.logger)
        self._image_uploader = _ImageUploader(self.logger)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """依次调用文件、扫描、摘要和上传组件处理 Markdown 图片。"""
        self.log_step("step_1", "读取 Markdown 内容并定位图片目录")
        md_content, md_path, image_dir = self._file_handler.read_md(state)

        self.log_step("step_2", "扫描有效图片并组装上下文")
        image_list = self._image_scanner.scan_img_dir(
            image_dir=image_dir,
            md_content=md_content,
            image_extensions=self.config.image_extensions,
            context_length=self.config.img_content_length,
        )

        self.log_step("step_3", "生成图片摘要")
        image_summaries = self._vlm_summarizer.summarize_all(
            document_title=md_path.stem,
            image_list=image_list,
            vl_model=self.config.vl_model,
            requests_per_minute=self.config.requests_per_minute,
        )

        self.log_step("step_4", "上传图片并替换 Markdown 图片引用")
        new_md_content = self._image_uploader.upload_and_replace(
            document_name=md_path.stem,
            md_content=md_content,
            image_summaries=image_summaries,
            image_list=image_list,
            minio_bucket=self.config.minio_bucket,
            minio_base_url=self.config.get_minio_base_url(),
        )

        self.log_step("step_5", "备份处理后的 Markdown 内容")
        new_md_path = self._file_handler.backup(md_path, new_md_content)

        state["md_content"] = new_md_content
        state["md_path"] = str(new_md_path)
        return state
