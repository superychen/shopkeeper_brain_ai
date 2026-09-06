"""PDF 转 Markdown 节点，仅 PDF 分支会经过这里。

EntryNode 设置 pdf_path → 校验输入和输出目录 → 启动 MinerU 子进程
→ 定位生成的 Markdown → 写入 state.md_path → MdToImgNode 处理图片。
本节点不直接生成向量；转换失败抛 PdfConversionError，中止后续节点。
"""

import os
import subprocess
import sys
from pathlib import Path

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.exceptions import (
    PdfConversionError,
    StateFieldError,
)
from knowledge.processor.import_processor.state import ImportGraphState


class PdfToMdNode(BaseNode):
    """将导入的 PDF 文件转换为 Markdown。"""

    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """校验路径并调用 MinerU 完成 PDF 转换。"""
        input_path, output_dir = self._validate_state(state)
        self.logger.info(
            "PDF 转换状态校验通过: input_path=%s, output_dir=%s",
            input_path,
            output_dir,
        )
        self._run_mineru(input_path, output_dir)

        md_path = self.get_md_url(input_path, output_dir)
        state["md_path"] = str(md_path)
        self.logger.info("Markdown 输出路径已写入状态: md_path=%s", md_path)
        return state

    def _run_mineru(self, input_path: Path, output_dir: Path) -> None:
        """启动 MinerU 子进程，并实时记录转换日志。"""
        command = [
            sys.executable,
            "-m",
            "mineru.cli.client",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-b",
            "pipeline",
            "--source=local",
        ]
        process_env = os.environ.copy()
        process_env["PYTHONIOENCODING"] = "utf-8"

        self.logger.info("开始执行 MinerU PDF 转换")
        try:
            with subprocess.Popen(
                # 要执行的命令及参数列表，列表形式可以避免额外的 Shell 解析。
                command,
                # 捕获标准输出，供当前节点逐行读取 MinerU 日志。
                stdout=subprocess.PIPE,
                # 将标准错误合并到标准输出，避免遗漏错误日志或双管道阻塞。
                stderr=subprocess.STDOUT,
                # 以字符串而不是字节形式读取进程输出。
                text=True,
                # 使用 UTF-8 解码输出，保证中文日志正常显示。
                encoding="utf-8",
                # 遇到无法解码的字符时用替代符处理，避免日志读取中断。
                errors="replace",
                # 在文本模式下按行缓冲，便于逐行记录实时日志。
                bufsize=1,
                # 继承当前环境变量，并传入前面设置的 UTF-8 输出配置。
                env=process_env,
            ) as process:
                if process.stdout is None:
                    raise PdfConversionError(
                        message="无法读取 MinerU 进程输出",
                        node_name=self.name,
                    )

                for line in process.stdout:
                    log_message = line.rstrip()
                    if log_message:
                        self.logger.info("[MinerU转换] %s", log_message)

                return_code = process.wait()
        except OSError as exc:
            raise PdfConversionError(
                message="无法启动 MinerU PDF 转换进程",
                node_name=self.name,
                cause=exc,
            ) from exc

        if return_code != 0:
            raise PdfConversionError(
                message=f"MinerU PDF 转换失败，退出码: {return_code}",
                node_name=self.name,
            )

        self.logger.info("MinerU PDF 转换完成")

    def get_md_url(self, input_path: Path, output_dir: Path) -> Path:
        """获取 MinerU 生成的 Markdown 文件完整路径。"""
        md_path = output_dir / input_path.stem / "auto" / f"{input_path.stem}.md"
        if not md_path.is_file():
            raise PdfConversionError(
                message=f"未找到 MinerU 生成的 Markdown 文件: {md_path}",
                node_name=self.name,
            )

        return md_path.resolve()

    def _validate_state(self, state: ImportGraphState) -> tuple[Path, Path]:
        """校验输入状态，返回 PDF 绝对路径及其所在的输出目录。"""
        if not isinstance(state, dict):
            raise StateFieldError(
                node_name=self.name,
                field_name="state",
                expected_type=dict,
            )

        import_file_path = state.get("import_file_path")
        if not isinstance(import_file_path, str) or not import_file_path.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
            )

        # 去除接口输入中可能出现的首尾空格，并允许用户目录写法（如 ~/docs/a.pdf）。
        input_path = Path(import_file_path.strip()).expanduser()
        if input_path.suffix.lower() != ".pdf":
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
                message=(
                    "状态字段 'import_file_path' 必须指向 PDF 文件，"
                    f"实际路径: {input_path}"
                ),
            )

        if not input_path.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
                message=f"状态字段 'import_file_path' 指向的文件不存在: {input_path}",
            )

        if not input_path.is_file():
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
                message=f"状态字段 'import_file_path' 必须指向普通文件: {input_path}",
            )

        # 统一为绝对路径，避免后续子进程因工作目录变化而找不到文件。
        input_path = input_path.resolve()
        return input_path, input_path.parent
