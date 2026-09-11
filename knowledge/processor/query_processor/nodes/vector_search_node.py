"""【04～05】目标商品＋原问题 → 双向量 → 同商品真实切片。

图入口调用继承的 RetrievalNode.__call__，再进入本文件 run → retrieve。
retrieve 将同步模型编码和数据库检索分别交给受控线程池，事件循环继续处理其他路。
例：标准商品 RS-12、问题“怎么测电压？” → 编码“目标商品：RS-12＋用户问题”
→ Milvus 同商品/profile 的混合召回 → embedding_chunks 和 vector_search_meta。
"""
from knowledge.processor.query_processor.retrieval_base import RetrievalNode
from knowledge.service.retrieval_runtime import ENCODING, MILVUS


class VectorSearchNode(RetrievalNode):
    """原问题召回路；其字段与 HyDE/网络分开，避免图并行更新冲突。"""
    route, output, meta = "vector", "embedding_chunks", "vector_search_meta"

    def __init__(self, config, embedding, repository):
        """注入配置、同步编码服务和正文仓储；依赖由图构建函数统一创建/共享。"""
        super().__init__(config)
        self.embedding, self.repository = embedding, repository

    async def retrieve(self, item, query):
        """单件商品完整检索，返回 (真实切片列表, 告警列表)。

        item 是库内标准名而非用户别名，query 保留完整问题及比较约束。
        例：item="RS-12"、query="怎么用？" → 编码器接收目标商品和用户问题两行；
        embed 返回 (EmbeddingVectors, profile)，随后仓储按这两个信息查询正文。
        self.route 用于标识候选来源；HyDE 子类复用本方法时该值为 hyde。
        编码或数据库失败交给基类商品隔离逻辑，不返回伪造的空成功结果。
        """
        # await 让出事件循环；实际 BGE 同步推理由 ENCODING 的单个工作线程执行。
        vectors, profile = await ENCODING.run(self.embedding.embed, f"目标商品：{item}\n用户问题：{query}")
        # profile 是入库模型空间指纹，不是相似度；item 是过滤范围，不由向量猜测。
        return await MILVUS.run(self.repository.search, vectors, profile, item, self.route)

    async def run(self, data, records, hits, warnings):
        """将每件商品交给基类 products；候选和诊断写入本路三个局部容器。

        例：items=[A,B] 将调用 retrieve(A,问题)、retrieve(B,问题)，每件最多5条，
        避免A的文档很多时挤掉B的全部候选。本方法不返回整个 state。
        """
        await self.products(data, records, hits, warnings, self.retrieve)
