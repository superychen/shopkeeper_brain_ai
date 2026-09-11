"""图中只保留查询结果与诊断，不存储大向量或共享可变默认列表。"""

from typing import TypedDict


class QueryGraphState(TypedDict, total=False):
    """贯穿【01～09】的共享数据包；TypedDict 仅提供类型提示，不做运行时验证。

    total=False 表示字段可分阶段出现，运行时输入约束由 QueryInput 校验。
    读表述和证据看 mention_results；看能否继续检索用 item_confirm_status/item_names。
    """

    request_id: str  # 基类生成/沿用，用于关联一次请求的所有步骤日志
    session_id: str  # 调用方提供的会话标识；本期没有自动查询历史的行为
    original_query: str  # 原问题保留，防止改写后丢掉原始约束
    history: list[dict]  # 可选上下文，如 [{"role": "user", "content": "我用 RS-12"}]
    item_confirm_status: str  # pending/confirmed/needs_clarification/not_found/error
    mention_results: list[dict]  # 每个表述的候选、分数、判定原因和证据
    item_names: list[str]  # 只有全部目标确认后才有值，是下游可用的库内商品范围
    confirmed_items: list[dict]  # 可有部分成功，携带名称、分数、命中文档来源
    options: list[dict]  # 按 mention_id 分组的展示候选，包含 has_more
    rewritten_query: str  # 成功时的商品名称绑定＋原问题；不是 LLM 提取草稿
    answer: str  # 澄清/无匹配/错误提示；不是产品知识答案，成功时为空
    error_code: str  # 正常业务结果为空；故障例如 milvus_unavailable
    warnings: list[str]  # 如 candidate_recall_truncated，不包含异常正文
    retrieval_input: dict  # 准备步骤写入，三路只读
    embedding_chunks: list[dict]  # 原问题真实切片，如 [{chunk_id:64位字符串, content:正文, route:vector, rank:1, score:0.6}]
    hyde_embedding_chunks: list[dict]  # 与上面结构相同、route=hyde；不存放模型生成的假设文本
    web_search_docs: list[dict]  # 网页证据，如 [{title:标题, url:链接, snippet:摘要, item_names:[A]}]
    vector_search_meta: dict  # 原问题路诊断，如 {status:partial, items:[商品级状态], warnings:[], elapsed_seconds:1.2}
    hyde_search_meta: dict  # HyDE路独占诊断字段；生成失败与无切片是不同状态
    web_search_meta: dict  # 网络路独占诊断字段；关闭为skipped，缺配置为failed
    retrieval_status: str  # join汇总：success/empty/partial/failed/skipped；有候选不等于生成了最终答案
    retrieval_warnings: list[str]  # join合并去重的告警和错误码，例如 [mcp_auth_failed]，不含异常原文


def empty_result() -> dict:
    """每次都创建新的列表并覆盖节点输出，避免状态残留。

    例：上一轮 item_names=["RS-12"]，下一轮 Milvus 失败，错误结果必须 item_names=[]。
    如果只写 error_code，LangGraph 合并状态时会保留旧名称，容易造成错误放行。
    """
    return dict(item_confirm_status="pending", mention_results=[], item_names=[],
                confirmed_items=[], options=[], rewritten_query="", answer="",
                error_code="", warnings=[])


def create_default_state(**overrides) -> QueryGraphState:
    """为单次图调用补默认状态；**overrides 最后展开，因此调用方提供的字段优先。

    例：create_default_state(original_query="RS-12 怎么用？") 创建独立 history/options 列表。
    自定义 overrides 中的对象不会深复制；调用方应为不同请求提供独立输入对象。
    """
    return {"request_id": "", "session_id": "", "original_query": "", "history": [],
            **empty_result(), **overrides}
