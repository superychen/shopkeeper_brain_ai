"""切片批量编码服务：把有序文本列表转换成有序双向量列表。

调用链：BgeEmbeddingChunksNode.process → embed → AIClients.get_bge_m3
        → 模型 encode_documents → embedding_vector_util.extract_vectors。

三个职责分别是：编码前检查完整文本的 token 数、确定实际模型版本、顺序分批推理。
本模块不访问 Milvus，也不修改图状态；成功时返回 (向量列表, 模型指纹)。
"""

import hashlib
import json
import logging
import time
from pathlib import Path

from knowledge.processor.import_processor.exceptions import EmbeddingError
from knowledge.service.embedding_vector_util import extract_vectors
from knowledge.utils.client.ai_clients import AIClients

logger = logging.getLogger("import.chunk_embedding")


def digest(value) -> str:
    """将可 JSON 序列化的数据转为64字符 SHA-256 摘要。

    sort_keys 固定字典键顺序，separators 去掉无关空格，UTF-8 固定中文编码方式。
    因而两个键顺序不同但内容相同的字典有相同摘要；allow_nan=False 拒绝非标准数值。
    这里生成的是可复现的内容标识，不是密码加密，也不能反推出原数据。
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def embedding_text(content: str, item_name: str) -> str:
    """返回模型实际接收的文本，同时供入库阶段重新计算摘要。

    有名称："RS-12\\n正文"；无名称："正文"。
    filter(None, ...) 去掉空字符串，join 把剩余两段用换行连接。
    strip 和 CRLF 规范化只作用于返回副本，原始正文仍供检索结果展示。
    """
    return "\n".join(filter(None, [item_name.strip(), content.replace("\r\n", "\n").strip()]))


class ChunkEmbeddingService:
    """本地推理服务。config 决定模型、设备、token 预算及批量大小。"""
    def __init__(self, config):
        """保存配置；昂贵的模型加载延迟到 embed 真正需要时才发生。"""
        self.config = config

    def _profile(self, client, tokenizer) -> str:
        """为实际权重、分词器和编码模板生成同一向量空间的指纹。

        输入 client 是 pymilvus 的 BGEM3EmbeddingFunction，内部逐层包装了
        FlagEmbedding 推理器、M3 推理模型和 Transformer 基础模型，因此需逐层取 config。
        优先使用 HuggingFace 权重提交号；本地模型无提交号时读取实际权重计算摘要。
        仅用模型名称或1024维度不足以保证兼容，权重换了也可能输出1024维。
        """
        model_config = client.model.model.model.config
        revision = getattr(model_config, "_commit_hash", None)
        if not revision:
            # 本地模型没有 HuggingFace commit，改为校验实际权重，不能只哈希目录名。
            root = Path(self.config.bge_m3_model_name)
            files = sorted(root.glob("*.safetensors")) or sorted(root.glob("pytorch_model*.bin"))
            if not files:
                raise EmbeddingError(message="无法确定模型权重版本，请使用缓存快照或本地模型目录")
            # .pt 文件还包含稀疏等投影层，漏掉它们会把不同 sparse 模型误当作同一版本。
            files += sorted(root.glob("*.pt"))
            checksum = hashlib.sha256()
            for path in files:
                checksum.update(path.name.encode("utf-8"))
                with path.open("rb") as source:
                    # iter(读取函数, 结束标记) 每次读取1MiB，遇到空字节结束，避免整份权重进内存。
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        checksum.update(block)
            revision = checksum.hexdigest()
        # tokenizer 改变会改变 token ID；模板改变会改变输入文本，两者都需要体现在指纹中。
        return digest({"weights": revision, "config": model_config.to_dict(),
                       "tokenizer": tokenizer.backend_tokenizer.to_str(),
                       "template": "item-newline-content:v1", "normalize": True})

    def embed(self, texts):
        """输入 texts，返回 ([(dense, sparse), ...], profile)，保持位置一一对应。

        处理顺序：取模型 → 检查所有文本长度 → 计算指纹 → 顺序分批编码 → 汇总返回。
        texts 已由节点规范化；本方法不再切正文。任何文本超长或批次失败都抛 EmbeddingError。
        """
        # 【流程 07.6 · 编码内部】本方法由 BgeEmbeddingChunksNode 调用；分批编码后按输入顺序返回全部向量。
        started = time.perf_counter()
        try:
            # Step 1：循环外只取一次单例，各批次共享模型权重和设备资源。
            client = AIClients.get_bge_m3(self.config)
            tokenizer = client.model.tokenizer
            # Step 2：取应用限制、tokenizer限制和BGE-M3上限中的最小值。
            # 字符数不能替代token数，且首尾特殊token也占位置；禁止依靠模型静默截断。
            limit = min(self.config.embedding_max_tokens, int(tokenizer.model_max_length), 8192)
            for index, text in enumerate(texts):
                size = len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
                if size > limit:
                    raise EmbeddingError(message=f"chunk[{index}] 超出 token 预算: actual={size}, limit={limit}")
            # Step 3：指纹随向量交给图，再由仓储拒绝不同向量空间混写。
            profile = self._profile(client, tokenizer)
            vectors = []
            # Step 4：例如20条、batch=8，start依次为0/8/16，切片长度为8/8/4。
            for start in range(0, len(texts), self.config.embedding_batch_size):
                batch = texts[start:start + self.config.embedding_batch_size]
                logger.info("切片编码批次: start=%d, count=%d, total=%d", start, len(batch), len(texts))
                # encode_documents 得到模型格式；extract_vectors 转成原生list/dict并校验数量和数值。
                # extend把本批多条结果逐个接到总列表；append会形成额外一层批次列表。
                vectors.extend(extract_vectors(client.encode_documents(batch), len(batch), self.config.embedding_dim))
            logger.info("切片编码全部完成: count=%d, elapsed=%.3fs", len(vectors), time.perf_counter() - started)
            return vectors, profile
        except EmbeddingError:
            # 已经带有业务含义的异常原样保留，调用方可区分长度、维度等原因。
            raise
        except Exception as exc:
            logger.error("切片编码失败: error_type=%s", type(exc).__name__)
            raise EmbeddingError(message="BGE-M3 切片编码失败", node_name="bge_embedding_chunks_node", cause=exc) from exc
