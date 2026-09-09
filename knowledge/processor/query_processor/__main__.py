"""【00 / 09】命令行入口与最终输出，建议从这里开始阅读。

运行示例（先在项目根目录激活 knowledge 的虚拟环境）：
    python -m knowledge.processor.query_processor "RS-12 怎么测电阻？"

调用链：main → run_item_name_confirm → 图.invoke → 节点.__call__
→ 基类.process → 子类._process → 提取/编码/检索/评分/汇总 → 返回 JSON。
程序调用者可以跳过本文件，直接使用 main_graph.py 的入口。
"""

import argparse
import json
import logging

from knowledge.processor.query_processor.main_graph import run_item_name_confirm


def main(argv=None):
    """将命令行问题转成图输入，最后把结果状态转成 JSON 和进程退出码。

    argv=None 时 argparse 读取真实命令行；测试可传 ["RS-12 怎么用？"]。
    正常反问也是一次成功执行，因此 needs_clarification/not_found 返回 0；
    模型或数据库出错返回 error 和退出码 1，供脚本判断是否需要重试。
    """
    # 【00.1】这里只接收原始问题；--session-id 仅携带标识，不会自动查询聊天记录。
    parser = argparse.ArgumentParser(description="商品名确认：BGE-M3 混合检索，确认门槛 0.7")
    parser.add_argument("query", help="用户原始问题")
    parser.add_argument("--session-id", default="")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    try:
        # 【00.2】此时还不知道商品名，输入只有问题和会话标识。
        # 下一站 main_graph.run_item_name_confirm 负责补默认状态并驱动节点执行。
        result = run_item_name_confirm({"original_query": args.query, "session_id": args.session_id})
    except (ValueError, TypeError):
        # 图/配置初始化阶段的这两类异常尚未进入节点错误边界，在 CLI 转成安全提示。
        print(json.dumps({"item_confirm_status": "error", "error_code": "invalid_config",
                          "answer": "查询配置不合法，请检查环境变量。"}, ensure_ascii=False))
        return 1
    # 【09】图已执行到 END。示例摘要：
    # {"item_confirm_status": "confirmed", "item_names": ["RS-12 数字万用表"], "answer": ""}
    # 这里只输出确认结果；真正的“测电阻操作步骤”要由后续正文检索和答案生成提供。
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["item_confirm_status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
