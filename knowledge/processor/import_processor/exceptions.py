"""
导入流程自定义异常类

统一错误处理，提供更清晰的错误信息

异常走向：服务/仓储抛出具体类型 → BaseNode 保留业务异常或包装未知异常
→ LangGraph 中止执行 → CLI 捕获 ImportProcessError 并返回退出码 1。
ConfigurationError 表示配置契约问题；EmbeddingError 表示编码问题；
MilvusError 表示写入或核验问题，调用方可按类型定位失败阶段。
"""


class ImportProcessError(Exception):
    """导入流程基础异常"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        """保存业务说明、发生位置和原异常；原异常仅留在对象中供程序诊断。"""
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        """格式化对外错误摘要，避免直接展开可能含请求或凭据的底层异常正文。"""
        parts = []
        if self.node_name:
            parts.append(f"[{self.node_name}]")
        parts.append(super().__str__())
        if self.cause:
            # SDK异常可能附带请求正文或鉴权信息，对外只显示异常类型。
            parts.append(f"(原因类型: {type(self.cause).__name__})")
        return " ".join(parts)



class StateFieldError(ImportProcessError):
    """状态字段错误。

    从 state 中获取必需字段缺失、为空或类型不符时抛出。

    Attributes:
        field_name: 缺失或无效的字段名称。
        expected_type: 期望的字段类型（可选）。
    """

    def __init__(
            self,
            node_name: str = "",
            field_name: str = "",
            expected_type: type = None,
            message: str = "",
            cause: Exception = None,
    ):
        # 调用方提供业务说明时优先保留；否则用字段名与期望类型生成定位信息。
        self.field_name = field_name
        self.expected_type = expected_type
        if not message:
            message = f"状态字段 '{field_name}' 缺失或无效"
            if expected_type:
                message += f"，期望类型: {expected_type.__name__}"
        super().__init__(message, node_name=node_name, cause=cause)


class ConfigurationError(ImportProcessError):
    """配置错误：环境变量缺失或配置值无效"""
    pass


class FileProcessingError(ImportProcessError):
    """文件处理错误：文件不存在、格式错误、读写失败"""
    pass


class PdfConversionError(FileProcessingError):
    """PDF 转换错误：MinerU 转换失败"""
    pass


class ImageProcessingError(FileProcessingError):
    """图片处理错误：图片总结、上传失败"""
    pass


class DocumentSplitError(ImportProcessError):
    """文档切分错误：切分逻辑异常"""
    pass


class EmbeddingError(ImportProcessError):
    """向量化错误：模型调用失败、向量生成异常"""
    pass


class LLMError(ImportProcessError):
    """LLM 调用错误：API 调用失败、响应解析失败"""
    pass


class StorageError(ImportProcessError):
    """存储错误：数据库操作失败"""
    pass


class MilvusError(StorageError):
    """Milvus 存储错误"""
    pass


class MinioError(StorageError):
    """MinIO 存储错误"""
    pass


class ValidationError(ImportProcessError):
    """数据验证错误：输入数据不符合预期"""
    pass
