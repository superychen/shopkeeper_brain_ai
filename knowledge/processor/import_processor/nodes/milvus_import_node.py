"""切片入库节点：导入主图的最后一个业务节点。

流程位置：BgeEmbeddingChunksNode → 本节点 → END。
数据走向：state.chunks → ChunkRepository.write → 已核验实体 → 新 chunks 和成功状态。
节点负责“向图交付结果”，仓储负责“怎样建集合、写数据、核验和清理旧切片”。
"""

import copy

from knowledge.processor.import_processor.base import BaseNode
from knowledge.service.chunk_repository import ChunkRepository


class MilvusImportNode(BaseNode):
    """把仓储的写入结果转换为 LangGraph 可合并的状态更新。"""
    name = "milvus_import_node"

    def __init__(self, config=None):
        """创建仓储对象但不连接数据库，保证构图本身没有网络副作用。"""
        super().__init__(config)
        self._repository = ChunkRepository(self.config)

    def process(self, state):
        """完整写入成功后回填切片主键，并将 import_status 改为 succeeded。

        输入必须来自向量化节点，包含 dense/sparse、编码摘要和编码数量。
        仓储若在任一步失败，此方法不会发布成功状态；但此前的数据库写入可能已发生，
        这不是跨批次事务。重试由稳定主键 upsert 保证不会不断增加重复记录。
        """
        # Step 1：write 内部完成所有写入和读回校验；返回顺序与输入切片一致。
        entities = self._repository.write(state)
        # Step 2：只修改副本，给调用方保留入库前状态用于诊断。
        chunks = copy.deepcopy(state["chunks"])
        # chunk_id 是数据库主键；record_hash 用来对账业务载荷，不是浮点向量摘要。
        for chunk, entity in zip(chunks, entities, strict=True):
            chunk.update(chunk_id=entity["chunk_id"], record_hash=entity["record_hash"])
        # Step 3：全部完成后一次返回。written_chunk_count 包括覆盖写，不等于新增条数。
        return {"chunks": chunks, "chunks_collection": self.config.chunks_collection,
                "milvus_chunk_ids": [row["chunk_id"] for row in entities],
                "written_chunk_count": len(entities), "import_status": "succeeded"}
