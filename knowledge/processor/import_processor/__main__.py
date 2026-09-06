"""命令行导入入口，可执行 python -m knowledge.processor.import_processor 文件路径。

流程：解析参数 → 初始化日志 → run_import_graph 串行执行所有节点
→ 打印任务/入库摘要 → 以退出码通知调用方成功或失败。
完整节点顺序请看 main_graph.py；本模块只负责终端参数与结果展示。
"""

import argparse
import json
import logging

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.exceptions import ImportProcessError
from knowledge.processor.import_processor.main_graph import run_import_graph


def main() -> int:
    """执行一次导入；成功返回 0，已知导入业务异常返回 1。"""
    # 第一步：把命令行输入转换成图需要的状态字段。
    # 同一业务文档更新时复用 document_id，仓储据此覆盖切片并清理旧尾部。
    parser = argparse.ArgumentParser(description="将 PDF/Markdown 串行导入本地 Milvus")
    parser.add_argument("path", help="PDF 或 Markdown 文件路径")
    parser.add_argument("--document-id", default="", help="重复导入时保持稳定的业务文档标识")
    args = parser.parse_args()
    setup_logging()
    try:
        # 第二步：同步调用图，直到最终入库节点完成或任意节点抛出异常。
        result = run_import_graph({"import_file_path": args.path, "document_id": args.document_id})
    except ImportProcessError as exc:
        logging.getLogger("import.cli").error("导入失败: node=%s, error_type=%s", exc.node_name, type(exc).__name__)
        return 1
    # 第三步：只输出摘要，不把原文、大体积向量或配置凭据打印到终端。
    # ensure_ascii=False 保留中文；字典推导式从最终状态中选择公开字段。
    print(json.dumps({key: result.get(key) for key in (
        "task_id", "document_id", "import_status", "written_chunk_count", "chunks_collection", "import_warnings"
    )}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # 模块被 import 时不自动执行；作为命令行运行时将返回值交给操作系统。
    raise SystemExit(main())
