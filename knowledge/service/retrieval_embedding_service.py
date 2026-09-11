"""【04】正文查询编码：保持入库模型指纹，拒绝静默截断和 sparse 再归一化。

VectorSearchNode.retrieve → 受控编码线程 → embed → 缓存BGE → 检查token预算
→ 复用文档profile算法 → encode_queries → extract_vectors → 返回向量和profile。
HyDE 同样走这条路径，只是输入多了一段假设。此服务不访问数据库或更新图状态。
"""
from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.service.chunk_embedding_service import ChunkEmbeddingService
from knowledge.service.embedding_vector_util import extract_vectors
from knowledge.service.item_name_embedding_service import EmbeddingVectors
from knowledge.utils.client.ai_clients import AIClients


class RetrievalEmbeddingService:
    """两路共享一个服务实例，缓存当前模型对象及其文档空间指纹。

    profile 必须和入库端算法相同；查询文本不等于文档模板，但不能因此产生另一种
    查询专属profile去过滤数据库，否则所有正常文档都会因指纹不匹配而被排除。
    """
    def __init__(self, config):
        """保存配置并初始化空缓存；模型直到第一次 embed 才加载。"""
        self.config = config
        self._model = None
        self._profile = None

    def embed(self, text):
        """编码一条查询文本，返回 (EmbeddingVectors, profile字符串)。

        输入示例：'目标商品：RS-12' 加换行和 '用户问题：怎么用？'。
        输出结构示例：EmbeddingVectors(dense=[0.1,...], sparse={12:0.7})、64位指纹；
        数值仅为演示权重，dense真实长度由 embedding_dim 决定，sparse键是token ID。
        模型不一致、输入超token预算、向量格式非法等统一抛 embedding_failed，
        交给商品级错误处理；不能截掉用户限制后继续检索，也不能跳过坏向量行。
        """
        try:
            cfg = self.config.shared
            model = AIClients.get_bge_m3(cfg)
            if getattr(model, "model_name", cfg.bge_m3_model_name) != cfg.bge_m3_model_name:
                raise ValueError("缓存模型不一致")
            tokenizer = model.model.tokenizer
            # 同时服从应用、tokenizer、BGE上限；中文字符数不等于token数，特殊token也计入。
            limit = min(cfg.embedding_max_tokens, int(tokenizer.model_max_length), 8192)
            if len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]) > limit:
                raise ValueError("查询超出 token 预算")
            if self._model is not model:
                # 用 is 比较模型实例身份：更换模型后重算指纹，同实例避免重复读取大权重文件。
                self._profile = ChunkEmbeddingService(cfg).profile(model, tokenizer)
                self._model = model
            # [text] 是一条查询构成的批次；extract_vectors 校验行数/维度/有限值，
            # [0] 取唯一的一行。稀疏权重沿用编码器结果，不额外做L2归一化。
            dense, sparse = extract_vectors(model.encode_queries([text]), 1, cfg.embedding_dim)[0]
            return EmbeddingVectors(dense, sparse), self._profile
        except Exception as exc:
            raise QueryError("embedding_failed", "正文查询编码失败", exc) from exc
