"""把 BGE 的不同输出格式统一成 Milvus 能接收的 Python 类型。

调用走向：编码服务 → extract_vectors（拆批次）→ validate_vectors（逐条校验）
→ 返回 [(稠密列表, 稀疏字典), ...] → 编码节点回填切片 → 仓储再次校验。
稠密向量表达整体语义；稀疏向量用 token ID 与权重表达词项特征。
任何一行非法都会中止整批，不能跳过坏行，否则向量会与原切片错位。
"""

import math
import struct
from numbers import Integral, Real

from knowledge.processor.import_processor.exceptions import EmbeddingError

FLOAT32_MAX = 3.4028234663852886e38


def _number(value) -> float:
    """接收 Python/NumPy 实数，转换成 float，并检查能否存入 float32。"""
    # bool 是 int 的子类，必须单独排除，避免 True 被当成向量权重 1。
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EmbeddingError(message="向量元素必须是数值")
    value = float(value)
    if not math.isfinite(value) or abs(value) > FLOAT32_MAX:
        raise EmbeddingError(message="向量包含非有限值或 float32 溢出")
    return value


def validate_vectors(dense, sparse, dim: int) -> tuple[list[float], dict[int, float]]:
    """校验一条切片的两种向量，返回新列表与新字典，不修改模型原始结果。

    dim 来自导入配置，必须与 Milvus schema 一致；失败抛出 EmbeddingError，
    由节点公共异常处理层记录失败并向上传播，终止后续导入。
    """
    # 第一步：固定稠密维度，并逐元素检查数值范围。
    if len(dense) != dim:
        raise EmbeddingError(message=f"dense 维度不匹配: expected={dim}, actual={len(dense)}")
    dense = [_number(value) for value in dense]
    # Python float 通常是双精度；先模拟 float32 转换，防止极小值入库后全变为 0。
    if not any(struct.unpack("f", struct.pack("f", value))[0] for value in dense):
        raise EmbeddingError(message="dense 不能是全零向量")
    if not isinstance(sparse, dict) or not sparse:
        raise EmbeddingError(message="sparse 必须是非空字典")
    # 第二步：把稀疏键统一成整数。JSON 常把整数键转换成字符串。
    normalized = {}
    for key, value in sparse.items():
        if isinstance(key, str) and key.isascii() and key.isdecimal():
            key = int(key)
        if isinstance(key, bool) or not isinstance(key, Integral) or not 0 <= key < 2**32 - 1:
            raise EmbeddingError(message="sparse token ID 非法")
        key = int(key)
        # 原字典可以同时含 1 和 "1"；转换后若覆盖，会悄悄丢失一个权重。
        if key in normalized:
            raise EmbeddingError(message="sparse token ID 重复")
        value = _number(value)
        # 权重在转换前后都必须为正，不能接受 float32 下溢后变成 0 的值。
        if value <= 0 or struct.unpack("f", struct.pack("f", value))[0] <= 0:
            raise EmbeddingError(message="sparse 权重必须为正数")
        normalized[key] = value
    return dense, normalized


def extract_vectors(result, expected_rows: int, dim: int):
    """按输入顺序提取整批向量，兼容 PyMilvus 与 FlagEmbedding 的字段名。

    expected_rows 是本批输入文本数。输出列表第 i 项只能对应输入第 i 条，
    所以稠密行数、稀疏行数及 CSR 行边界都必须严格匹配。
    """
    if not isinstance(result, dict):
        raise EmbeddingError(message="模型返回值必须是字典")
    # 两套编码接口的命名不同，在此收敛，节点和仓储不需要知道底层差异。
    dense = result.get("dense", result.get("dense_vecs"))
    sparse = result.get("sparse", result.get("lexical_weights"))
    if dense is None or len(dense) != expected_rows or sparse is None:
        raise EmbeddingError(message="模型输出缺失或 dense 行数不匹配")
    if hasattr(sparse, "indptr"):
        # CSR 用三个数组压缩存储：indptr 给出每行边界，indices 是 token ID，
        # data 是对应权重。例如 indptr=[0,2,3] 表示第一行取 [0:2]，第二行取 [2:3]。
        ptr, ids, values = sparse.indptr, sparse.indices, sparse.data
        if (len(ptr) != expected_rows + 1 or ptr[0] != 0
                or any(not isinstance(value, Integral) for value in ptr)
                or ptr[-1] != len(ids) or len(ids) != len(values)
                or any(a > b for a, b in zip(ptr, ptr[1:]))):
            raise EmbeddingError(message="CSR 行指针或非零数组不一致")
        rows = []
        for i in range(expected_rows):
            # 切片右端不包含在内；strict=True 拒绝长度不等，避免 zip 静默截断。
            pairs = list(zip(ids[ptr[i]:ptr[i + 1]], values[ptr[i]:ptr[i + 1]], strict=True))
            if len({key for key, _ in pairs}) != len(pairs):
                raise EmbeddingError(message="CSR 含重复 token ID")
            # CSR 中显式存储的 0 没有词项贡献，去除后再执行正权重校验。
            rows.append({key: value for key, value in pairs if value != 0})
    else:
        # 单条结果可能直接是字典，多条结果必须是按输入顺序排列的字典列表。
        rows = [sparse] if isinstance(sparse, dict) and expected_rows == 1 else sparse
        if not isinstance(rows, (list, tuple)) or len(rows) != expected_rows:
            raise EmbeddingError(message="sparse 行数不匹配")
    # 最后逐行配对校验；列表推导式生成的新列表就是编码服务的统一返回值。
    return [validate_vectors(d, s, dim) for d, s in zip(dense, rows, strict=True)]
