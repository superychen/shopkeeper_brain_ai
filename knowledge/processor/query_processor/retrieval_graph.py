"""【01～08】完整检索入口：确认 → 准备 → 三路并行 → 汇总 → END。

异步示例：await arun_query_retrieval({"original_query": "RS-12 怎么用？"})。
这里只返回候选；RRF、重排和最终答案由后续开发阶段负责。

阅读路线：arun_query_retrieval → build_retrieval_graph → reset → 已有确认节点
→ prepare → 三个 RetrievalNode.__call__ 并行执行 → join → END。
例如问“RS-12 怎么用”：确认成功后，三路分别寻找原问题切片、HyDE 切片、网页摘要；
如果确认节点要求用户选择商品，流程直接结束，不产生三路外部调用。
OUTPUTS 保存可供后续使用的证据，METAS 保存执行状态；“查到了什么”和“查得是否完整”
必须同时返回。网络失败时仍可保留两路本地证据，但汇总状态应为 partial。
"""
from langgraph.graph import START, END, StateGraph
from knowledge.processor.query_processor.state import QueryGraphState, create_default_state
from knowledge.processor.query_processor.main_graph import route_after_item_confirm
from knowledge.processor.query_processor.retrieval_config import RetrievalConfig
from knowledge.processor.query_processor.nodes.item_name_confirm_node import ItemNameConfirmNode
from knowledge.processor.query_processor.nodes.vector_search_node import VectorSearchNode
from knowledge.processor.query_processor.nodes.hyde_search_node import HyDESearchNode
from knowledge.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode

# 顺序固定为向量、HyDE、网络；每一路只拥有对应的一组输出/诊断字段。
OUTPUTS = ("embedding_chunks", "hyde_embedding_chunks", "web_search_docs")
METAS = ("vector_search_meta", "hyde_search_meta", "web_search_meta")


def reset_retrieval():
    """【01.1】创建本轮检索初始值，不修改商品确认字段。

    返回：三份空候选列表、三份 skipped 诊断、空检索输入和汇总告警。
    例：上一轮 embedding_chunks=[切片A]，本轮商品尚待澄清，必须返回 []，
    否则 LangGraph 合并状态时可能把旧切片显示为本轮证据。
    每个推导式都新建列表/字典，不在多个请求之间共用可变默认容器。
    """
    return {**{k: [] for k in OUTPUTS}, **{k: {"status": "skipped", "items": [], "warnings": []} for k in METAS},
            "retrieval_input": {}, "retrieval_status": "skipped", "retrieval_warnings": []}


def prepare(state):
    """【02】把可信确认结果转换成三路共用输入，失败时阻止 fan-out。

    输入示例：item_names=["RS-12", "RS-12"]，original_query="怎么用？"。
    输出示例：retrieval_input={"items": ["RS-12"], "query": "怎么用？",
    "request_id": 当前请求ID}，retrieval_status="pending"。
    dict.fromkeys 按首次出现顺序去重，不改大小写或库内名称，精确过滤依赖原值。
    空名称、超过5件商品、非字符串问题等返回 failed 和 invalid_retrieval_input；
    此时检索输入为空，图的条件边会到 END。这里不访问模型/Milvus，避免本地
    依赖故障发生在分叉之前、连带阻断本可独立运行的网络节点。
    """
    names, query = state.get("item_names"), state.get("original_query")
    if (not isinstance(names, list) or not 1 <= len(names) <= 5
            or any(not isinstance(n, str) or not n.strip() or len(n) > 1024 for n in names)
            or not isinstance(query, str) or not query.strip() or len(query) > 2000):
        return reset_retrieval() | {"retrieval_status": "failed", "retrieval_warnings": ["invalid_retrieval_input"]}
    return reset_retrieval() | {"retrieval_input": {"items": list(dict.fromkeys(names)),
        "query": query, "request_id": state.get("request_id", "")}, "retrieval_status": "pending"}


def join(state):
    """【08】等三路都进入终态后，统一判断完整性；不融合或重排候选。

    输入：三份候选列表和三份 meta；返回 retrieval_status/retrieval_warnings。
    示例（按向量/HyDE/网络顺序）：
      success + success + failed → partial，保留两路已有切片；
      empty + empty + skipped → empty，网络关闭不算失败；
      failed + failed + failed → failed，不能解释为知识库无资料；
      empty + failed + skipped → partial，即使总候选为零仍是不完整检索。
    告警合并各路 warnings 和商品/节点 error_code，集合去重后排序便于稳定展示。
    """
    metas = [state[k] for k in METAS]
    active = [m for m in metas if m["status"] != "skipped"]
    statuses = [m["status"] for m in active]
    if not statuses:
        status = "skipped"
    elif all(s == "failed" for s in statuses):
        status = "failed"
    elif any(s in {"failed", "partial"} for s in statuses):
        status = "partial"
    else:
        status = "success" if any(state[k] for k in OUTPUTS) else "empty"
    warnings = {w for m in metas for w in m.get("warnings", [])}
    warnings.update(r["error_code"] for m in metas for r in m.get("items", []) if r.get("error_code"))
    return {"retrieval_status": status, "retrieval_warnings": sorted(warnings)}


def build_retrieval_graph(config=None, *, confirm_node=None, nodes=None):
    """【01.2】组装并编译图，返回可重复 ainvoke 的流程对象。

    config 为 RetrievalConfig；confirm_node 可替换确认节点；nodes 必须按
    [向量节点, HyDE节点, 网络节点] 提供三个可调用对象，主要用于离线测试。
    例：graph=build_retrieval_graph(cfg)，随后 await graph.ainvoke(本轮输入)。
    compile 只建立调度关系，不执行 LLM/Milvus/MCP 请求。
    两路本地节点共享 embedding/repository 实例，使模型 profile 和集合就绪检查
    可以复用；每次新建图则这些实例缓存重新开始，适合脚本但不适合高频请求。
    """
    cfg = config or RetrievalConfig()
    if nodes is None:
        from knowledge.service.retrieval_embedding_service import RetrievalEmbeddingService
        from knowledge.service.chunk_search_repository import ChunkSearchRepository
        from knowledge.service.hyde_generation_service import HyDEGenerationService
        from knowledge.service.bailian_search_service import BailianSearchService
        embedding, repository = RetrievalEmbeddingService(cfg), ChunkSearchRepository(cfg)
        nodes = [VectorSearchNode(cfg, embedding, repository),
                 HyDESearchNode(cfg, embedding, repository, HyDEGenerationService(cfg)),
                 WebMcpSearchNode(cfg, BailianSearchService(cfg))]
    if len(nodes) != 3:
        raise ValueError("必须提供三路节点")
    graph = StateGraph(QueryGraphState)
    graph.add_node("reset", lambda state: reset_retrieval())
    graph.add_node("confirm", confirm_node if confirm_node is not None else ItemNameConfirmNode(cfg.query))
    graph.add_node("prepare", prepare)
    graph.add_node("join", join)
    graph.add_edge(START, "reset")
    graph.add_edge("reset", "confirm")
    # 【01.3】只有整体 confirmed 且有标准商品名才放行；部分确认也要等待澄清。
    graph.add_conditional_edges("confirm", route_after_item_confirm, {"retrieve": "prepare", "end": END})
    routes = ["vector_search", "hyde_search", "web_mcp_search"]
    for name, node in zip(routes, nodes):
        graph.add_node(name, node)
    # 条件分支返回列表表示 fan-out；列表前驱的单条边构成等待全部完成的屏障。
    graph.add_conditional_edges("prepare", lambda s: routes if s["retrieval_input"] else END,
                                {**{n: n for n in routes}, END: END})
    graph.add_edge(routes, "join")
    # 三个前驱放在同一条列表边上，join 等全部完成；不是谁先结束就先汇总。
    graph.add_edge("join", END)
    return graph.compile()


async def arun_query_retrieval(state, config=None, **dependencies):
    """【01】提交一次完整检索，返回保留确认结果和三路结果的最终图状态。

    调用例：await arun_query_retrieval({"original_query": "RS-12 怎么用？"})。
    返回例（演示）：item_confirm_status="confirmed"，retrieval_status="partial"，
    embedding_chunks=[真实切片]，web_search_docs=[]，网络 meta 说明失败原因。
    只放行原问题、request_id、session_id、history 四个请求字段；即使调用方
    传 item_confirm_status="confirmed" 也会被丢弃，必须重新经过确认节点。
    **dependencies 将测试替身按关键字传入构图函数，不是用户可提交的业务字段。
    异步服务直接 await；仅同步脚本最外层使用 asyncio.run，不能在已有事件循环
    内再次创建循环。本函数不承诺最终知识答案，answer 仍遵循确认节点原有语义。
    """
    inputs = {k: state[k] for k in ("original_query", "request_id", "session_id", "history") if k in state}
    return await build_retrieval_graph(config, **dependencies).ainvoke(create_default_state(**inputs))
