"""编码与入库共用的数据契约：先检查本地输入，再调用模型或数据库。

调用顺序：validate_chunks → validate_config → 文档字段 → 逐条切片字段。
这里仅验证，不修正、不重排切片；错误应由上游生产节点修复，不能在入库时
静默补默认值。验证失败抛出业务异常，由公共节点层记录失败并停止后续步骤。
"""

import math
import re

from knowledge.processor.import_processor.exceptions import ConfigurationError, StateFieldError


def validate_config(config):
    """检查批量、容量、超时与集合名；成功无返回值，失败直接抛异常。"""
    # 使用 type(...) is int 是为了拒绝 bool；批量大小等参数不能把 True 当成 1。
    for field in ("embedding_dim", "embedding_batch_size", "embedding_max_tokens", "max_import_chunks",
                  "milvus_batch_size", "milvus_max_batch_bytes"):
        value = getattr(config, field)
        if type(value) is not int or value <= 0:
            raise ConfigurationError(message=f"{field} 必须是正整数")
    if type(config.milvus_max_retries) is not int or config.milvus_max_retries < 0:
        raise ConfigurationError(message="milvus_max_retries 必须是非负整数")
    # 重试可以为 0；超时必须为有限正数，不能是 NaN、无穷或布尔值。
    timeout = config.milvus_timeout_seconds
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ConfigurationError(message="milvus_timeout_seconds 必须是有限正数")
    # 集合名会参与数据库操作，提前限定字符范围，避免服务调用后才发现名称非法。
    for field in ("chunks_collection", "item_name_collection"):
        value = getattr(config, field)
        if not isinstance(value, str) or len(value) > 255 or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ConfigurationError(message=f"{field} 集合名非法")
    # 商品名集合与切片集合的 schema 不同，不能共用同一集合。
    if config.chunks_collection == config.item_name_collection:
        raise ConfigurationError(message="切片与商品名必须使用不同集合")


def text_field(value, name, limit, *, empty=False):
    """检查一个文本字段并原样返回；limit 是 UTF-8 字节数，不是字符数。

    星号后的 empty 只能通过关键字传入，使调用处清楚表达是否允许空文本。
    name 用于定位具体的切片字段；NUL 字符不允许进入持久化数据。
    """
    if not isinstance(value, str) or "\0" in value or (not empty and not value.strip()) or len(value.encode("utf-8")) > limit:
        raise StateFieldError(field_name=name, message=f"{name} 类型、内容或 UTF-8 长度非法")
    return value


def validate_chunks(state, config):
    """验证文档及全部切片，返回原 chunks 列表供调用方读取。

    本函数不复制数据。需要回填向量或主键的节点应自行 deepcopy，避免原地
    修改 LangGraph 输入状态，导致失败重试时混入上一次运行的中间结果。
    """
    # 第一步：检查配置与文档级边界，限制单任务数量，避免无限批量占用资源。
    validate_config(config)
    if not isinstance(state, dict):
        raise StateFieldError(field_name="state", expected_type=dict)
    text_field(state.get("document_id"), "document_id", 128)
    chunks = state.get("chunks")
    if not isinstance(chunks, list) or not chunks or len(chunks) > config.max_import_chunks:
        raise StateFieldError(field_name="chunks", message="chunks 必须非空且不超过任务上限")
    item_name = text_field(state.get("item_name", ""), "item_name", 1024, empty=True)
    # 第二步：enumerate 同时取得位置和切片；序号必须从 0 连续排列。
    # 稳定主键和旧尾部清理都依赖该规则，不能把乱序输入直接写入数据库。
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or type(chunk.get("chunk_index")) is not int or chunk["chunk_index"] != index:
            raise StateFieldError(field_name=f"chunks[{index}]", message="切片必须按从0开始的连续序号排列")
        # 第三步：字段大小与数据库 VARCHAR 约束保持一致；正文/父标题允许为空。
        for field, limit in (("content", 65535), ("file_title", 1024), ("title", 4096),
                             ("parent_title", 4096), ("body", 65535), ("source_path", 8192)):
            text_field(chunk.get(field), f"chunks[{index}].{field}", limit, empty=field in {"body", "parent_title"})
        if chunk.get("item_name", "") != item_name:
            raise StateFieldError(field_name=f"chunks[{index}].item_name", message="切片与文档商品名不一致")
        # 第四步：保留标题路径和来源章节的类型，供检索后的溯源展示使用。
        for field, kind in (("heading_path", str), ("source_titles", str), ("source_section_indexes", int)):
            values = chunk.get(field)
            if not isinstance(values, list) or any(type(v) is not kind for v in values):
                raise StateFieldError(field_name=f"chunks[{index}].{field}")
            if kind is int and any(v < 0 for v in values):
                raise StateFieldError(field_name=f"chunks[{index}].{field}")
        # char_count 统计 Python 字符数；它与前面的数据库字节长度是两种指标。
        if type(chunk.get("char_count")) is not int or chunk["char_count"] != len(chunk["content"]):
            raise StateFieldError(field_name=f"chunks[{index}].char_count")
    return chunks
