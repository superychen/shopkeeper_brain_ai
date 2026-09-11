"""【06】假设仅扩展检索输入；输出仍是 Milvus 中真实存在的切片。

调用路线：基类 __call__ → 继承的 run/products → 本类 retrieve → generator.generate
→ 父类 retrieve 编码/查库 → hyde_embedding_chunks 与 hyde_search_meta。
区别在于查询文本增加假设段落，数据库范围、向量格式和融合算法与原问题路一致。
"""
from knowledge.processor.query_processor.nodes.vector_search_node import VectorSearchNode


class HyDESearchNode(VectorSearchNode):
    """复用向量检索流程，只替换查询准备步骤和输出归属，避免复制两套仓储逻辑。"""
    route, output, meta = "hyde", "hyde_embedding_chunks", "hyde_search_meta"

    def __init__(self, config, embedding, repository, generator):
        """除共用编码/仓储外注入生成器；测试可让生成器返回固定文本或主动失败。"""
        super().__init__(config, embedding, repository)
        self.generator = generator

    async def retrieve(self, item, query):
        """先生成检索假设，再用“原问题＋假设”召回真实正文。

        演示：query="怎么测电压？"，假设为“查找档位、接线、额定范围说明”；
        父类接收到原问题和带“非事实”标记的假设文本，返回真实切片而非该假设。
        super().retrieve 调用父类算法，但其中 self.route 仍为 hyde，所以输出
        证据的 route 不会错写成 vector。hypothetical 仅局部使用，不写入图状态。
        生成异常直接向上传播；不能用原问题再查一次并冒充独立 HyDE 结果。
        """
        # 生成失败直接传播到商品级错误处理，不能复制原问题召回冒充 HyDE。
        hypothetical = await self.generator.generate(item, query)
        return await super().retrieve(item, query + "\n检索假设（非事实）：" + hypothetical)
