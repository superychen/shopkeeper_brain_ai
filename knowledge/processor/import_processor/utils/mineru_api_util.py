"""通过 MinerU API 将 PDF 转换为 Markdown。"""

import logging
from pathlib import Path

import httpx

from knowledge.processor.import_processor.exceptions import PdfConversionError


class MinerUApiUtil:
    """调用已启动的 MinerU API，并只保存 Markdown 结果。"""

    name = "mineru_api_util"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 3600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(f"import.{self.name}")

    def convert_pdf_to_md(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
    ) -> Path:
        """调用 MinerU API 转换 PDF，并返回本地 Markdown 完整路径。"""
        input_path = Path(pdf_path).expanduser().resolve()
        target_dir = Path(output_dir).expanduser().resolve()
        self._validate_input(input_path)

        # 只请求 Markdown，避免返回中间 JSON、模型结果、图片和原始 PDF。
        form_data = {
            "parse_method": "auto",
            "return_md": "true",
            "return_middle_json": "false",
            "return_model_output": "false",
            "return_content_list": "false",
            "return_images": "false",
            "response_format_zip": "false",
            "return_original_file": "false",
            "client_side_output_generation": "false",
        }
        endpoint = f"{self.base_url}/file_parse"

        self.logger.info("开始调用 MinerU API 转换 PDF: file_name=%s", input_path.name)
        try:
            with input_path.open("rb") as pdf_file:
                files = {
                    "files": (input_path.name, pdf_file, "application/pdf"),
                }
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(endpoint, data=form_data, files=files)
                    response.raise_for_status()
                    response_data = response.json()
        except (OSError, httpx.HTTPError, ValueError) as exc:
            self.logger.error(
                "MinerU API 调用失败: file_name=%s, error=%s",
                input_path.name,
                exc,
            )
            raise PdfConversionError(
                message="MinerU API PDF 转换失败",
                node_name=self.name,
                cause=exc,
            ) from exc

        md_content = self._get_md_content(response_data)
        md_path = target_dir / f"{input_path.stem}.md"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md_content, encoding="utf-8")
        except OSError as exc:
            self.logger.error(
                "保存 Markdown 文件失败: md_path=%s, error=%s",
                md_path,
                exc,
            )
            raise PdfConversionError(
                message=f"保存 Markdown 文件失败: {md_path}",
                node_name=self.name,
                cause=exc,
            ) from exc

        self.logger.info("MinerU API 转换完成: md_path=%s", md_path)
        return md_path

    def _validate_input(self, input_path: Path) -> None:
        """校验待转换的 PDF 文件。"""
        if input_path.suffix.lower() != ".pdf" or not input_path.is_file():
            raise PdfConversionError(
                message=f"PDF 文件不存在或格式无效: {input_path}",
                node_name=self.name,
            )

    def _get_md_content(self, response_data: object) -> str:
        """从 MinerU 单文件转换响应中提取 Markdown 内容。"""
        if not isinstance(response_data, dict):
            raise PdfConversionError(
                message="MinerU API 返回格式无效",
                node_name=self.name,
            )

        results = response_data.get("results")
        if not isinstance(results, dict) or len(results) != 1:
            raise PdfConversionError(
                message="MinerU API 未返回唯一的 PDF 转换结果",
                node_name=self.name,
            )

        result = next(iter(results.values()))
        md_content = result.get("md_content") if isinstance(result, dict) else None
        if not isinstance(md_content, str):
            raise PdfConversionError(
                message="MinerU API 响应中缺少 Markdown 内容",
                node_name=self.name,
            )

        return md_content


def main() -> None:
    """使用 tmp_dir 中的示例 PDF 验证 PdfToMdNode 转换方法。"""
    # 仅示例入口依赖节点，保持 MinerUApiUtil 正常导入时的独立性。
    from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    import_processor_dir = Path(__file__).resolve().parent.parent
    pdf_path = import_processor_dir / "tmp_dir" / "万用表的使用.pdf"
    output_dir = import_processor_dir / "tmp_dir" / "mineru_output"

    node = PdfToMdNode()
    node._run_mineru(pdf_path, output_dir)
    md_path = node.get_md_url(pdf_path, output_dir)
    logging.getLogger(f"import.{node.name}").info(
        "示例 PDF 转换成功: md_path=%s",
        md_path,
    )


if __name__ == "__main__":
    main()
