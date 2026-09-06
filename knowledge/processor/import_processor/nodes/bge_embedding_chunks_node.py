"""切片向量化节点：把“带商品名的文本切片”变成“带双向量的切片”。

在主图中的位置：
    ItemNameRecognitionNode → 本节点 → MilvusImportNode

阅读顺序：process → validate_chunks → embedding_text → ChunkEmbeddingService.embed。
这里管理图状态；模型如何分批调用、CSR 如何解包分别封装在 service 和 util 中。
本节点不写 Milvus。只有所有批次编码成功，才返回包含新 chunks 的状态更新。
"""

import copy
import hashlib

from knowledge.processor.import_processor.base import BaseNode
from knowledge.service.chunk_embedding_service import ChunkEmbeddingService, digest, embedding_text
from knowledge.service.import_validation import validate_chunks


class BgeEmbeddingChunksNode(BaseNode):
    """连接 LangGraph 状态与批量编码服务，保持切片顺序和原始正文。"""
    name = "bge_embedding_chunks_node"

    def __init__(self, config=None):
        """复用基类配置/日志并创建轻量服务对象；此时还没有加载模型。"""
        super().__init__(config)
        self._service = ChunkEmbeddingService(self.config)

    def process(self, state):
        """执行“校验 → 文本准备 → 全量编码 → 副本回填 → 发布更新”。

        输入：包含 document_id、chunks 及可选 item_name 的 ImportGraphState。
        输出：状态更新字典，包括新 chunks、模型指纹、切分指纹、编码数量。
        异常：字段错误或任一编码批次失败时直接向上抛出，不交给入库节点。

        注意返回的是字典，不是切片列表；LangGraph 会把这些字段合并回整张图的状态。
        """
        # Step 1：整份文档先通过结构校验，避免编码几批后才发现后面的切片损坏。
        chunks = validate_chunks(state, self.config)
        # Step 2：列表推导式保持原顺序。商品名只加入编码副本，不覆盖 chunk.content。
        texts = [embedding_text(chunk["content"], state.get("item_name", "")) for chunk in chunks]
        # Step 3：一次服务调用内部执行多批编码，返回 [(dense, sparse), ...] 和模型指纹。
        vectors, profile = self._service.embed(texts)
        # Step 4：深拷贝使嵌套列表也与输入隔离，失败时调用方不会看到半更新切片。
        result = copy.deepcopy(chunks)
        # 三个序列按位置一一对应；strict=True 会在长度不等时失败，而不是截断丢片。
        for chunk, text, (dense, sparse) in zip(result, texts, vectors, strict=True):
            # 丢弃调用方可能遗留的主键，入库核验完成后才重新回填。
            chunk.pop("chunk_id", None)
            chunk.pop("record_hash", None)
            chunk.update(dense_vector=dense, sparse_vector=sparse,
                         embedding_text_hash=hashlib.sha256(text.encode("utf-8")).hexdigest())
        self.logger.info("切片向量化完成: document_id=%s, count=%d", state["document_id"], len(result))
        # Step 5：running 表示只有编码完成，不能提前宣布入库成功。
        # embedding_profile 标识实际模型；split_profile 标识切分策略和长度参数。
        # 下游仓储会用 embedding_text_hash 检查正文有没有在编码后被人修改。
        return {"chunks": result, "embedding_model": self.config.bge_m3_model_name,
                "embedding_profile": profile, "embedded_chunk_count": len(result),
                "import_status": "running", "written_chunk_count": 0, "milvus_chunk_ids": [],
                "split_profile": digest({"version": "split:v1", "max": self.config.max_content_length,
                                         "min": self.config.min_content_length, "overlap": self.config.overlap_sentences})}
