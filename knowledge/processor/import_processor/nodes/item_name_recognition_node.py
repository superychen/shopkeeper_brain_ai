"""商品名称识别节点，负责串联 LLM、向量服务、Milvus 与图状态。

典型示例：文档前部出现“RS PRO”“RS-12”“数字万用表”时，DeepSeek 先返回
``RS PRO RS-12 数字万用表`` 及原文证据；随后本地 BGE-M3 生成稠密/稀疏向量，
Milvus upsert 成功后，节点才把名称和主键一次性回填到 state 及所有 chunks。
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal, cast

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from knowledge.processor.import_processor.base import BaseNode
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    LLMError,
    StateFieldError,
)
from knowledge.processor.import_processor.state import ChunkRecord, ImportGraphState
from knowledge.prompt.import_prompt import (
    ITEM_NAME_SYSTEM_PROMPT,
    ITEM_NAME_USER_PROMPT_TEMPLATE,
)
from knowledge.schema.item_name import ItemNameExtraction
from knowledge.service.item_name_embedding_service import ItemNameEmbeddingService
from knowledge.service.item_name_repository import ItemNameRepository
from knowledge.utils.client.ai_clients import AIClients

RecognitionStatus = Literal["recognized", "not_found", "ambiguous"]


@dataclass(frozen=True, slots=True)
class ItemNameInputs:
    """保存校验后的节点输入，避免后续步骤反复读取不可信的原始 state。

    使用不可变 dataclass 表达“已通过入口校验”的数据边界；后续步骤拿到该对象后，
    可以直接使用 ``document_id`` 等字段，不必在每一层重复处理 None 和错误类型。
    """

    document_id: str
    file_title: str
    chunks: list[ChunkRecord]
    chunk_limit: int
    context_size: int
    collection_name: str
    task_id: str


@dataclass(frozen=True, slots=True)
class ItemNameRecognitionResult:
    """DeepSeek 输出经过本地业务校验后的结构。"""

    status: RecognitionStatus
    item_name: str
    brand: str
    model: str
    product_type: str
    confidence: float
    evidence: tuple[str, ...]


class ItemNameRecognitionNode(BaseNode):
    """识别唯一主商品并将混合向量幂等写入 Milvus。"""

    name = "item_name_recognition_node"
    _collection_name_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _generic_item_names = frozenset(
        {"产品", "商品", "设备", "仪器", "说明书", "用户手册", "操作指南"}
    )
    _quote_pairs = (("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’"))

    def __init__(self, config: ImportConfig | None = None) -> None:
        super().__init__(config=config)
        self._recognition_chain: (
                Runnable[dict[str, str], ItemNameExtraction] | None
        ) = None
        self._embedding_service = ItemNameEmbeddingService(self.config)
        self._repository = ItemNameRepository(self.config)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """按识别、向量化、持久化、回填的顺序完成商品名称节点。

        顺序是业务一致性约束：如果 BGE-M3 或 Milvus 失败，state 不应提前出现
        ``recognized`` 成功态；任务重试时会继续使用相同 document_id 和 Milvus 主键。
        """
        inputs = self._validate_inputs(state)
        context = self._build_recognition_context(inputs)
        self.logger.info(
            "开始识别商品名称: task_id=%s, document_id=%s, file_title=%s, "
            "context_chars=%d",
            inputs.task_id,
            inputs.document_id,
            inputs.file_title,
            len(context),
        )

        result = self._recognize_with_deepseek(inputs.file_title, context)
        if result.status != "recognized":
            # not_found/ambiguous 是合法业务结果，不应浪费本地模型计算或产生脏库记录。
            # 例如同时介绍两个并列型号、无法判断主次时，只回填 ambiguous 状态。
            self._fill_unrecognized_state(state, result)
            log_method = (
                self.logger.warning
                if result.status == "ambiguous"
                else self.logger.info
            )
            log_method(
                "商品名称未进入向量库: document_id=%s, status=%s",
                inputs.document_id,
                result.status,
            )
            return state

        vectors = self._embedding_service.embed(result.item_name)
        item_pk = self._repository.upsert(
            collection_name=inputs.collection_name,
            document_id=inputs.document_id,
            file_title=inputs.file_title,
            item_name=result.item_name,
            vectors=vectors,
        )

        # 成功态后置，外部调用中途失败时不会留下只回填了一半的 chunks。
        self._fill_recognized_state(state, result, item_pk)
        self.logger.info(
            "商品名称节点处理完成: document_id=%s, item_name=%s, pk=%s",
            inputs.document_id,
            result.item_name,
            item_pk[:12],
        )
        return state

    def _validate_inputs(self, state: ImportGraphState) -> ItemNameInputs:
        """校验 state 和本节点必需配置。"""
        if not isinstance(state, dict):
            raise StateFieldError(
                node_name=self.name,
                field_name="state",
                expected_type=dict,
            )

        document_id = self._require_state_text(state, "document_id")
        file_title = self._require_state_text(state, "file_title")
        if len(document_id.encode("utf-8")) > 128:
            raise StateFieldError(
                node_name=self.name,
                field_name="document_id",
                expected_type=str,
                message="document_id 的 UTF-8 长度不能超过 128 字节",
            )
        if len(file_title.encode("utf-8")) > 1024:
            raise StateFieldError(
                node_name=self.name,
                field_name="file_title",
                expected_type=str,
                message="file_title 的 UTF-8 长度不能超过 1024 字节",
            )

        chunks_value = state.get("chunks")
        if not isinstance(chunks_value, list) or not chunks_value:
            raise StateFieldError(
                node_name=self.name,
                field_name="chunks",
                expected_type=list,
            )
        for index, chunk in enumerate(chunks_value):
            if not isinstance(chunk, dict):
                raise StateFieldError(
                    node_name=self.name,
                    field_name=f"chunks[{index}]",
                    expected_type=dict,
                )
            content = chunk.get("content")
            if not isinstance(content, str) or not content.strip():
                raise StateFieldError(
                    node_name=self.name,
                    field_name=f"chunks[{index}].content",
                    expected_type=str,
                )

        chunk_limit = self._require_positive_int(
            self.config.item_name_chunk_k,
            "item_name_chunk_k",
        )
        context_size = self._require_positive_int(
            self.config.item_name_chunk_size,
            "item_name_chunk_size",
        )
        if context_size < 100:
            raise ConfigurationError(
                message="item_name_chunk_size 不能小于 100",
                node_name=self.name,
            )
        self._require_positive_int(self.config.embedding_dim, "embedding_dim")

        if not isinstance(self.config.deepseek_timeout_seconds, Real) or isinstance(
                self.config.deepseek_timeout_seconds,
                bool,
        ) or self.config.deepseek_timeout_seconds <= 0:
            raise ConfigurationError(
                message="deepseek_timeout_seconds 必须是正数",
                node_name=self.name,
            )
        retries = self.config.deepseek_max_retries
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 0:
            raise ConfigurationError(
                message="deepseek_max_retries 必须是非负整数",
                node_name=self.name,
            )
        self._require_config_text("deepseek_llm_model")
        embedding_model = self._require_config_text("bge_m3_model_name")
        if len(embedding_model.encode("utf-8")) > 128:
            raise ConfigurationError(
                message="bge_m3_model_name 的 UTF-8 长度不能超过 128 字节",
                node_name=self.name,
            )

        collection_name = self._require_config_text("item_name_collection")
        if not self._collection_name_pattern.fullmatch(collection_name):
            raise ConfigurationError(
                message="item_name_collection 不是合法的 Milvus 集合名",
                node_name=self.name,
            )

        task_id_value = state.get("task_id", "")
        task_id = task_id_value if isinstance(task_id_value, str) else ""
        return ItemNameInputs(
            document_id=document_id,
            file_title=file_title,
            chunks=cast(list[ChunkRecord], chunks_value),
            chunk_limit=chunk_limit,
            context_size=context_size,
            collection_name=collection_name,
            task_id=task_id,
        )

    def _build_recognition_context(self, inputs: ItemNameInputs) -> str:
        """按完整 chunk 优先构造限长上下文，极端首块超长时使用前缀兜底。

        例如预算 2500 字符时，前两个完整 chunk 合计 2100 字符、第三个加入后会达到
        2800 字符，则只提交前两个，避免在型号或表格中间截断并误导 LLM。
        """
        selected: list[str] = []
        used_chars = 0
        for chunk in inputs.chunks[: inputs.chunk_limit]:
            chunk_index = chunk.get("chunk_index", len(selected))
            title = str(chunk.get("title", "")).strip()
            block = (
                f'【切片 index="{chunk_index}" title="{title}"】\n'
                f'{chunk["content"]}\n'
                "【切片结束】"
            )
            separator_length = 2 if selected else 0
            if used_chars + separator_length + len(block) <= inputs.context_size:
                selected.append(block)
                used_chars += separator_length + len(block)
                continue

            if selected:
                break

            marker = "\n[上下文已截断]"
            prefix_size = max(1, inputs.context_size - len(marker))
            selected.append(block[:prefix_size] + marker)
            self.logger.warning(
                "首个商品名识别切片超过上下文预算，已仅截取识别副本: "
                "document_id=%s, chunk_index=%s, original_chars=%d, budget=%d",
                inputs.document_id,
                chunk_index,
                len(block),
                inputs.context_size,
            )
            break

        return "\n\n".join(selected)

    def _recognize_with_deepseek(
            self,
            file_title: str,
            context: str,
    ) -> ItemNameRecognitionResult:
        """通过 LangChain 结构化调用 DeepSeek，再验证原文证据。"""
        started_at = time.perf_counter()
        try:
            extraction = self._get_recognition_chain().invoke(
                {"file_title": file_title, "context": context}
            )
        except Exception as exc:
            self.logger.error(
                "LangChain DeepSeek 商品名识别失败: model=%s, error_type=%s",
                self.config.deepseek_llm_model,
                type(exc).__name__,
            )
            raise LLMError(
                message="LangChain DeepSeek 商品名识别失败",
                node_name=self.name,
                cause=exc,
            ) from exc
        if not isinstance(extraction, ItemNameExtraction):
            raise LLMError(
                message="LangChain 未返回 ItemNameExtraction 结构",
                node_name=self.name,
            )

        result = self._validate_extraction(
            extraction,
            source_text=f"{file_title}\n{context}",
        )
        self.logger.info(
            "LangChain DeepSeek 商品名识别完成: model=%s, status=%s, elapsed=%.3fs",
            self.config.deepseek_llm_model,
            result.status,
            time.perf_counter() - started_at,
        )
        if result.status == "recognized" and result.confidence < 0.5:
            self.logger.warning(
                "商品名称证据置信度较低: item_name=%s, confidence=%.3f",
                result.item_name,
                result.confidence,
            )
        return result

    def _get_recognition_chain(
            self,
    ) -> Runnable[dict[str, str], ItemNameExtraction]:
        """惰性创建可复用的 Prompt → DeepSeek → Pydantic 链。

        ``invoke`` 的输出已经是 ItemNameExtraction，而不是待手工解析的 JSON 字符串。
        链只在节点实例第一次使用时创建，后续文档复用同一个 ChatDeepSeek 客户端。
        """
        if self._recognition_chain is None:
            # System Prompt 包含 JSON 示例的大括号。使用 SystemMessage 固定内容，可避免
            # ChatPromptTemplate 把这些大括号误认为 {file_title} 一类模板变量。
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                    ("human", ITEM_NAME_USER_PROMPT_TEMPLATE),
                ]
            )
            structured_llm = AIClients.get_llm(self.config).with_structured_output(
                ItemNameExtraction,
                method="json_mode",
            )
            self._recognition_chain = cast(
                Runnable[dict[str, str], ItemNameExtraction],
                prompt | structured_llm,
            )
        return self._recognition_chain

    def _validate_extraction(
            self,
            extraction: ItemNameExtraction,
            *,
            source_text: str,
    ) -> ItemNameRecognitionResult:
        """补充 JSON Schema 无法表达的原文证据和业务语义校验。

        Pydantic 负责“字段是否合法”，这里负责“内容是否可信”。例如模型返回型号
        ``RS-99`` 虽然类型正确，但原文只有 ``RS-12``，证据校验仍会拒绝该结果。
        """
        normalized_fields = {
            key: self._normalize_short_text(getattr(extraction, key))
            for key in ("item_name", "brand", "model", "product_type")
        }
        evidence = tuple(item.strip() for item in extraction.evidence)

        if extraction.status != "recognized":
            return ItemNameRecognitionResult(
                status=extraction.status,
                item_name="",
                brand="",
                model="",
                product_type="",
                confidence=extraction.confidence,
                evidence=(),
            )

        item_name = normalized_fields["item_name"]
        if not item_name or len(item_name) > 256:
            raise LLMError(
                message="识别出的 item_name 为空或超过 256 个字符",
                node_name=self.name,
            )
        if item_name.casefold() in self._generic_item_names:
            raise LLMError(
                message="识别出的 item_name 是无效泛称",
                node_name=self.name,
            )
        if "```" in item_name or item_name.startswith(("答案", "商品名称")):
            raise LLMError(
                message="识别出的 item_name 含解释性或 Markdown 内容",
                node_name=self.name,
            )
        if any(item not in source_text for item in evidence):
            raise LLMError(
                message="DeepSeek evidence 不是输入中的原文",
                node_name=self.name,
            )

        normalized_source = unicodedata.normalize("NFKC", source_text).casefold()
        core_fields = (
            normalized_fields["brand"],
            normalized_fields["model"],
            normalized_fields["product_type"],
        )
        # any(...) 表示品牌、型号、商品类型至少命中一个；无需写三组重复的 if/else。
        if not any(
                field and unicodedata.normalize("NFKC", field).casefold() in normalized_source
                for field in core_fields
        ):
            raise LLMError(
                message="商品名称的品牌、型号或类型均无法在输入中找到",
                node_name=self.name,
            )

        return ItemNameRecognitionResult(
            status="recognized",
            item_name=item_name,
            brand=normalized_fields["brand"],
            model=normalized_fields["model"],
            product_type=normalized_fields["product_type"],
            confidence=extraction.confidence,
            evidence=evidence,
        )

    @staticmethod
    def _fill_unrecognized_state(
            state: ImportGraphState,
            result: ItemNameRecognitionResult,
    ) -> None:
        """回填有效的未识别业务结果，不给 chunks 写入空商品名。"""
        state.update(
            item_name="",
            item_name_status=result.status,
            item_name_confidence=result.confidence,
            item_name_evidence=[],
            item_name_milvus_pk="",
        )

    @staticmethod
    def _fill_recognized_state(
            state: ImportGraphState,
            result: ItemNameRecognitionResult,
            item_pk: str,
    ) -> None:
        """Milvus 成功后一次性发布商品名和新的 chunks 列表。

        ``{**chunk, ...}`` 会复制每个字典；例如原始 chunk 不带 item_name，返回的新
        chunk 带该字段，但调用方仍持有的旧对象不会被原地污染，失败时也便于回滚。
        """
        chunks = cast(list[ChunkRecord], state["chunks"])
        updated_chunks = [
            cast(ChunkRecord, {**chunk, "item_name": result.item_name})
            for chunk in chunks
        ]
        state.update(
            item_name=result.item_name,
            item_name_status="recognized",
            item_name_confidence=result.confidence,
            item_name_evidence=list(result.evidence),
            item_name_milvus_pk=item_pk,
            chunks=updated_chunks,
        )

    @staticmethod
    def _require_state_text(state: ImportGraphState, field_name: str) -> str:
        value = state.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise StateFieldError(
                node_name=ItemNameRecognitionNode.name,
                field_name=field_name,
                expected_type=str,
            )
        return value.strip()

    @staticmethod
    def _require_positive_int(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigurationError(
                message=f"{field_name} 必须是正整数",
                node_name=ItemNameRecognitionNode.name,
            )
        return value

    def _require_config_text(self, field_name: str) -> str:
        value = getattr(self.config, field_name, "")
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(
                message=f"缺少必需配置: {field_name}",
                node_name=self.name,
            )
        return value.strip()

    def _normalize_short_text(self, value: str) -> str:
        normalized = " ".join(unicodedata.normalize("NFKC", value).split())
        for opening, closing in self._quote_pairs:
            if normalized.startswith(opening) and normalized.endswith(closing):
                normalized = normalized[len(opening): -len(closing)].strip()
                break
        return normalized
