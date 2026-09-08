"""切片仓储：把已向量化的切片可靠地写入 Milvus。

对外入口只有 write(state)，由 MilvusImportNode.process 调用。按以下顺序阅读：
    prepare：校验全部切片，转换成数据库实体并计算主键
      → batches：按数量和字节预算分批
      → ensure_collection：创建或验证集合及索引
      → _call(upsert)：顺序写入每批
      → _call(get)：按主键读回并对账
      → _call(delete/query)：删除旧尾部并核验最终条数
      → 返回已核验实体，交给节点回填成功状态。

关键边界：Python中的state不是数据库事务。第2批失败时，第1批可能已经写入。
因此采用稳定主键upsert支持重试，而不假设异常会自动撤销此前的外部写入。
"""

import hashlib
import json
import logging
import time

import grpc
from pymilvus import DataType

from knowledge.processor.import_processor.exceptions import ConfigurationError, MilvusError
from knowledge.service.chunk_embedding_service import digest, embedding_text
from knowledge.service.embedding_vector_util import validate_vectors
from knowledge.service.import_validation import text_field, validate_chunks
from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger("import.chunk_repository")
# 单一字段清单同时服务建表和入库校验，长度单位为UTF-8字节，中文不等于一个字节。
# 文档关联字段用于过滤；profile用于防止模型混写；两个hash分别检查编码输入与业务载荷。
TEXT_FIELDS = {"chunk_id": 64, "document_id": 128, "content": 65535, "file_title": 1024,
               "title": 4096, "parent_title": 4096, "item_name": 1024, "item_name_milvus_pk": 64,
               "embedding_model": 128, "embedding_profile": 128, "split_profile": 128,
               "embedding_text_hash": 64, "record_hash": 64}
# 较复杂的来源信息集中放入JSON字段，不把任意state字段直接传给数据库。
META_FIELDS = ("body", "heading_path", "source_titles", "source_section_indexes", "char_count", "source_path")


class ChunkRepository:
    """持久化层，只关心切片如何入库，不负责抽取商品名或调用大模型。"""
    def __init__(self, config):
        """保存同一份导入配置，真正的数据库连接在write中延迟取得。"""
        self.config = config

    def _call(self, method, **kwargs):
        """包装一次Milvus操作，统一超时、有限重试、日志与异常分类。

        method是可调用方法，例如client.upsert；**kwargs把关键字参数字典展开给它。
        返回值保留SDK原格式，是否成功由上层按操作类型检查，不能仅凭请求返回判断。
        网络暂时断开可以重试，字段/配置错误不能靠反复请求修复。
        """
        # max_retries=2表示“首次+两次重试”，因此循环上限要加1。
        for attempt in range(self.config.milvus_max_retries + 1):
            try:
                return method(timeout=self.config.milvus_timeout_seconds, **kwargs)
            except (ConfigurationError, MilvusError):
                raise
            except Exception as exc:
                transient = isinstance(exc, (TimeoutError, ConnectionError)) or (
                    isinstance(exc, grpc.RpcError) and exc.code() in {
                        grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED})
                if transient and attempt < self.config.milvus_max_retries:
                    logger.warning("Milvus 瞬时故障，准备重试: attempt=%d, error_type=%s", attempt + 1, type(exc).__name__)
                    # 退避时间依次0.5、1、2秒并封顶，避免服务故障时连续密集请求。
                    time.sleep(min(0.5 * 2**attempt, 2))
                    continue
                logger.error("Milvus 操作失败: error_type=%s", type(exc).__name__)
                # from exc保留原始异常链供排查，用户日志只记录类型，避免SDK正文泄漏。
                raise MilvusError(message="Milvus 操作失败", node_name="milvus_import_node", cause=exc) from exc

    def ensure_collection(self, client):
        """确保目标集合可接收本版实体；成功无返回值，不兼容则抛ConfigurationError。

        集合可类比关系数据库的表；Schema定义字段，索引规定向量检索方式。
        不存在：构建Schema和双索引后创建；已存在：同样做完整校验，不能直接跳过。
        本方法不删除旧集合、不自动迁移类型，避免意外覆盖用户已有知识库。
        """
        name = self.config.chunks_collection
        # Step 1：应用自己生成主键，所以关闭auto_id；关闭动态字段让拼错字段尽早暴露。
        if not self._call(client.has_collection, collection_name=name):
            schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
            for field, length in TEXT_FIELDS.items():
                schema.add_field(field_name=field, datatype=DataType.VARCHAR,
                                 max_length=length, is_primary=field == "chunk_id")
            # 序号用于有序还原和旧尾部清理；metadata保留完整标题路径及来源信息。
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
            schema.add_field(field_name="updated_at", datatype=DataType.INT64)
            schema.add_field(field_name="metadata", datatype=DataType.JSON)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=self.config.embedding_dim)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            # Step 2：dense用COSINE比较语义方向；sparse用IP累加重合token的权重乘积。
            # 两个索引分别建立，并不意味着导入时已经执行混合检索或排序。
            indexes = client.prepare_index_params()
            indexes.add_index(field_name="dense_vector", index_name="dense_vector_index", index_type="AUTOINDEX", metric_type="COSINE")
            indexes.add_index(field_name="sparse_vector", index_name="sparse_vector_index", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            try:
                self._call(client.create_collection, collection_name=name, schema=schema, index_params=indexes)
            except MilvusError:
                # 其他执行者可能抢先建表，复查后仍须完整验证，不能直接吞掉异常。
                if not self._call(client.has_collection, collection_name=name):
                    raise
            logger.info("切片集合创建完成: collection=%s", name)
        # Step 3：无论新建还是复用都读服务端实际Schema，不能只相信本地配置。
        description = self._call(client.describe_collection, collection_name=name)
        if description.get("auto_id") is not False or description.get("enable_dynamic_field") is not False:
            raise ConfigurationError(message="切片集合必须关闭 auto_id 和动态字段")
        # 以字段名构建字典后，校验不依赖SDK返回的字段排列顺序。
        fields = {field["name"]: field for field in description.get("fields", [])}
        expected = {**{key: DataType.VARCHAR for key in TEXT_FIELDS}, "chunk_index": DataType.INT64,
                    "updated_at": DataType.INT64, "metadata": DataType.JSON,
                    "dense_vector": DataType.FLOAT_VECTOR, "sparse_vector": DataType.SPARSE_FLOAT_VECTOR}
        # dict转set只比较键；多字段和少字段都拒绝，保持持久化契约明确。
        if set(fields) != set(expected):
            raise ConfigurationError(message="切片集合字段集合不兼容，请新建集合迁移")
        for key, datatype in expected.items():
            actual = fields[key]
            if int(actual.get("type", -1)) != int(datatype) or bool(actual.get("is_primary")) != (key == "chunk_id"):
                raise ConfigurationError(message=f"切片集合字段类型或主键不兼容: {key}")
            if key in TEXT_FIELDS and int(actual.get("params", {}).get("max_length", -1)) != TEXT_FIELDS[key]:
                raise ConfigurationError(message=f"切片集合字段长度不兼容: {key}")
        if int(fields["dense_vector"].get("params", {}).get("dim", -1)) != self.config.embedding_dim:
            raise ConfigurationError(message="切片集合向量维度不兼容")
        # Step 4：有向量字段不代表已有正确索引；检查索引名称、所属字段和度量。
        names = self._call(client.list_indexes, collection_name=name)
        for field, metric in (("dense_vector", "COSINE"), ("sparse_vector", "IP")):
            index_name = f"{field}_index"
            if index_name not in names:
                raise ConfigurationError(message=f"切片集合缺少索引: {index_name}")
            index = self._call(client.describe_index, collection_name=name, index_name=index_name)
            actual_metric = index.get("metric_type", index.get("params", {}).get("metric_type"))
            if index.get("field_name") != field or actual_metric != metric:
                raise ConfigurationError(message=f"切片集合索引不兼容: {index_name}")

    def prepare(self, state):
        """输入图状态，返回可直接upsert的实体列表；整个方法不访问数据库。

        chunk保留程序处理信息，entity是符合Milvus Schema的一行，两者字段并不完全相同。
        所有实体准备完才允许写库，防止写了前几片才发现后面一片没有向量或字段超长。
        """
        # Step 1：验证切片结构和编码总数，确认当前模型配置没有在两节点之间改变。
        chunks = validate_chunks(state, self.config)
        if state.get("embedded_chunk_count") != len(chunks):
            raise MilvusError(message="向量化数量与切片数量不一致")
        for key in ("embedding_model", "embedding_profile", "split_profile"):
            text_field(state.get(key), key, TEXT_FIELDS[key])
        if state["embedding_model"] != self.config.bge_m3_model_name:
            raise ConfigurationError(message="状态模型与当前配置不一致")
        entities = []
        for chunk in chunks:
            # Step 2：重新计算实际编码文本摘要。若有人改了正文但保留旧向量，立即拒绝。
            expected_hash = hashlib.sha256(embedding_text(chunk["content"], state.get("item_name", "")).encode("utf-8")).hexdigest()
            if chunk.get("embedding_text_hash") != expected_hash:
                raise MilvusError(message=f"chunk[{chunk['chunk_index']}] 文本已变化，必须重新向量化")
            # 即使上游校验过也要守住写库边界：该节点也可能被单独调用或从中间文件恢复。
            dense, sparse = validate_vectors(chunk.get("dense_vector", []), chunk.get("sparse_vector"), self.config.embedding_dim)
            # Step 3：先拣选切片字段，再附加文档级字段，最后组装嵌套来源元数据。
            entity = {field: chunk[field] for field in ("content", "file_title", "title", "parent_title", "chunk_index")}
            entity.update({field: state[field] for field in ("document_id", "embedding_model", "embedding_profile", "split_profile")})
            entity.update(item_name=state.get("item_name", ""), item_name_milvus_pk=state.get("item_name_milvus_pk", ""),
                          embedding_text_hash=chunk.get("embedding_text_hash", ""),
                          metadata={key: chunk[key] for key in META_FIELDS})
            # JSON 元数据允许补充归档关联，不必修改已有集合 schema。
            if state.get("source_archive"):
                entity["metadata"]["source_archive"] = state["source_archive"]
            # Step 4：使用与建表一致的字节限制；metadata另设32KiB应用上限。
            for key, value in entity.items():
                if key in TEXT_FIELDS:
                    text_field(value, key, TEXT_FIELDS[key], empty=key in {"item_name", "item_name_milvus_pk", "parent_title"})
            if len(json.dumps(entity["metadata"], ensure_ascii=False).encode("utf-8")) > 32768:
                raise MilvusError(message=f"chunk[{chunk['chunk_index']}] 元数据超过32KiB")
            # Step 5：record_hash只覆盖此时的业务字段，不包含时间、主键和浮点向量。
            # 这样重复导入时即使时间变化，仍能通过同一业务载荷摘要核验。
            entity["record_hash"] = digest(entity)
            # 主键只由版本、document_id、chunk_index决定；\0分隔避免拼接歧义。
            # 名称或正文更新不会产生新主键，而会覆盖原序号对应的记录。
            entity["chunk_id"] = hashlib.sha256(f"chunk:v1\0{state['document_id']}\0{chunk['chunk_index']}".encode()).hexdigest()
            entity.update(dense_vector=dense, sparse_vector=sparse, updated_at=int(time.time() * 1000))
            entities.append(entity)
        return entities

    def batches(self, entities):
        """把实体列表拆成有序批次列表，例如5条、每批2条得到[2, 2, 1]。

        同时限制单批记录数和估算字节数。JSON大小加256字节是应用侧保守预算，
        不等于精确的protobuf网络包长度。单实体超限直接失败，无法靠继续分批解决。
        此方法先构造所有批次，因此超限错误在首次写库前就能发现。
        """
        batches, batch, size = [], [], 0
        for entity in entities:
            cost = len(json.dumps(entity, ensure_ascii=False, allow_nan=False).encode("utf-8")) + 256
            if cost > self.config.milvus_max_batch_bytes:
                raise MilvusError(message="单条切片超过 Milvus 请求预算")
            if batch and (len(batch) >= self.config.milvus_batch_size or size + cost > self.config.milvus_max_batch_bytes):
                # 先封存已有批次，再把当前实体放入新批次，当前实体不会被遗漏。
                batches.append(batch)
                batch, size = [], 0
            batch.append(entity)
            size += cost
        if batch:
            # 最后一批通常未达到上限，也必须交付，否则会漏掉末尾切片。
            batches.append(batch)
        return batches

    def write(self, state):
        """执行完整入库流程，返回与输入同序的已核验实体列表。

        输入为向量化节点输出；成功表示每条实体写入、读回及最终数量均已核对。
        任一阶段失败都抛异常；调用者不能把“部分批次已成功”解释为整份导入成功。
        外层run_import_graph持有单进程导入锁；直接调用仓储时由调用者保证互斥。
        """
        # 【流程 07.7 · 入库内部】先校验全部实体，再逐批写入；全部读回核验后才清旧尾部并检查总数。
        # 返回 MilvusImportNode 回填主键；任一步异常都交给门面处理，不在这里宣布 API 任务成功。
        # Step 1：先做所有本地校验、实体映射与分批，再连接外部数据库。
        entities = self.prepare(state)
        batches = self.batches(entities)
        client = StorageClients.get_milvus(self.config)
        name = self.config.chunks_collection
        # Step 2：准备集合和索引；load使集合能被后续查询/读回使用。
        self.ensure_collection(client)
        self._call(client.load_collection, collection_name=name)
        # 同维度不等于同模型；任何历史行的模型指纹不同都禁止混写。
        # Step 3：只需找到一条指纹不同的历史记录即可拒绝混写，因此limit=1足够。
        mismatch = self._call(client.query, collection_name=name,
                              filter=f"embedding_profile != {json.dumps(state['embedding_profile'])}",
                              output_fields=["chunk_id"], limit=1, consistency_level="Strong")
        if mismatch:
            raise ConfigurationError(message="集合包含其他 embedding 模型版本，请使用新集合")
        # Step 4：顺序upsert并检查回执。upsert_count包括覆盖更新，不代表新增数量。
        for index, batch in enumerate(batches):
            result = self._call(client.upsert, collection_name=name, data=batch)
            if not isinstance(result, dict) or result.get("upsert_count") != len(batch):
                raise MilvusError(message=f"批次{index} upsert 数量不一致")
            logger.info("切片写入批次完成: document_id=%s, batch=%d, count=%d", state["document_id"], index, len(batch))
        # Step 5：Strong读回本次主键，核对文档、序号及载荷摘要，而不依赖返回顺序。
        for batch in batches:
            rows = self._call(client.get, collection_name=name, ids=[row["chunk_id"] for row in batch],
                              output_fields=["chunk_id", "document_id", "chunk_index", "record_hash"], consistency_level="Strong")
            # 主键→三元组的映射对比能发现漏片、串文档和写入内容不一致。
            # 额外比较行数可发现重复行，避免字典覆盖重复键后掩盖问题。
            actual = {row["chunk_id"]: (row["document_id"], row["chunk_index"], row["record_hash"]) for row in rows}
            expected = {row["chunk_id"]: (row["document_id"], row["chunk_index"], row["record_hash"]) for row in batch}
            if actual != expected or len(rows) != len(batch):
                raise MilvusError(message="切片写后读回不一致")
        # Step 6：确认所有新片后再清旧尾部。例如原5片变3片，只删同文档index>=3。
        # json.dumps给字符串加引号并转义特殊字符，避免把业务ID直接拼成表达式语法。
        document_filter = f"document_id == {json.dumps(state['document_id'], ensure_ascii=True)}"
        self._call(client.delete, collection_name=name, filter=f"{document_filter} and chunk_index >= {len(entities)}")
        # Step 7：最后按文档精确计数，不能用集合全局条数推断本次结果。
        counts = self._call(client.query, collection_name=name, filter=document_filter,
                            output_fields=["count(*)"], consistency_level="Strong")
        if len(counts) != 1 or counts[0].get("count(*)") != len(entities):
            raise MilvusError(message="文档最终切片总数不一致")
        logger.info("切片入库及核验完成: document_id=%s, count=%d", state["document_id"], len(entities))
        # Step 8：交回节点后才会把主键和succeeded写入图状态。
        return entities
