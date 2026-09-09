"""查询失败保留内部原因，对外仅暴露稳定错误码和安全提示。"""


class QueryError(Exception):
    """服务层 → 节点基类的错误契约。

    例：QueryError("milvus_unavailable", "商品名称检索服务暂时不可用", timeout_error)。
    code 用于程序分支，message 用于安全提示，cause 只保留供内部诊断。
    QueryBaseNode.process 将它转换成 error 状态，不把数据库故障当成商品无匹配。
    """
    def __init__(self, code: str, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.cause = cause
