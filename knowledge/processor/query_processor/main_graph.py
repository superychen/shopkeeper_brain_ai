"""【01】查询图入口：先组装流程，再提交状态执行。

当前图：START → item_name_confirm → END，只有一个业务节点。
该节点内部的完整步骤位于 nodes/item_name_confirm_node.py，不能把“一个图节点”
误解为“只调用一次模型”。节点会依次提取、编码、检索，并对每个商品独立评分。

单次调用：run_item_name_confirm({"original_query": "RS-12 怎么用？"})。
常驻服务：graph = build_query_graph()，然后反复 graph.invoke(每次的新输入)。
前者每次构建图，后者可复用节点持有的仓储就绪缓存。
"""

from langgraph.graph import END, START, StateGraph

from knowledge.processor.query_processor.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_processor.state import QueryGraphState, create_default_state


def route_after_item_confirm(state: QueryGraphState) -> str:
    """给未来完整查询图预留的路由函数；当前最小图尚未挂载此条件边。

    confirmed 且 item_names 非空 → retrieve；其他情况 → end。
    例如 confirmed_items 已有 A、但 B 尚待确认，整体状态仍是澄清，不能进入检索。
    """
    return "retrieve" if state.get("item_confirm_status") == "confirmed" and state.get("item_names") else "end"


def build_query_graph(config=None, *, node=None):
    """定义并编译图，不在 compile 时调用 LLM 或 Milvus。

    node 参数用于注入带假服务的节点做测试；星号表示只能用 node=... 传参。
    真正开始计算的是返回对象的 invoke，而不是 add_node/add_edge。
    """
    # 【01.1】StateGraph 声明节点共享哪些状态字段；字段定义见 state.py。
    workflow = StateGraph(QueryGraphState)
    node = node if node is not None else ItemNameConfirmNode(config)
    workflow.add_node("item_name_confirm", node)
    # 【01.2】LangGraph 接收到可调用实例后，会执行 QueryBaseNode.__call__(state)。
    # 无论确认、反问还是 error 状态，本期都在节点结束后到 END，不执行正文检索。
    workflow.add_edge(START, "item_name_confirm")
    workflow.add_edge("item_name_confirm", END)
    return workflow.compile()


def run_item_name_confirm(state: QueryGraphState, config=None, *, node=None) -> QueryGraphState:
    """【01.3】补默认值并 invoke，返回合并了输入和节点输出的最终图状态。

    输入：{"original_query": "RS-12 怎么用？"}。
    默认补入：history=[]、item_names=[]、answer=""、item_confirm_status="pending" 等。
    节点返回状态更新后，LangGraph 将其合并进图状态；original_query 因此仍被保留。
    """
    return build_query_graph(config, node=node).invoke(create_default_state(**state))
