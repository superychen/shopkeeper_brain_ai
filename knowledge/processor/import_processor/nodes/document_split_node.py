"""文档切分节点：标题切分、长度治理以及最终 chunk 组装。"""

import re
from dataclasses import dataclass, replace
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    DocumentSplitError,
    StateFieldError,
)
from knowledge.processor.import_processor.state import ChunkRecord, ImportGraphState
from knowledge.processor.import_processor.utils.markdown_table_util import (
    MarkdownTableUtil,
)


@dataclass(frozen=True, slots=True)
class SplitInputs:
    """保存 Step 1 校验、规范化后的输入，供后续步骤统一使用。"""

    md_content: str
    file_title: str
    md_path: Path
    file_dir: Path
    max_content_length: int
    min_content_length: int
    overlap_sentences: int


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """表示按 Markdown 标题得到的一个语义章节。

    ``heading_path`` 保存从最高层有效祖先到当前标题的完整路径。例如 H1
    直接跳到 H3 时，路径是 ``("# 第一章", "### 参数")``，H2 缺失不会
    阻止 H3 找到 H1 作为父标题。
    """

    title: str
    heading_level: int
    parent_title: str
    heading_path: tuple[str, ...]
    body: str
    source_section_index: int


@dataclass(frozen=True, slots=True)
class _FenceMarker:
    """记录已打开代码围栏的字符类型和长度。"""

    character: str
    length: int


@dataclass(frozen=True, slots=True)
class _MarkdownBlock:
    """Step 3 的内部文本块；atomic=True 表示不应从块内部切开。"""

    text: str
    kind: str
    atomic: bool


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """Step 3 结果：一个或多个 section 片段组成的待组装切片。

    示例：``## 参数`` 正文过长时会先产生多个仅含一个 part 的 draft；若相邻
    两个短 section 可以安全合并，则一个 draft 会包含两个 part。Step 4 再把它
    转成可序列化的 :class:`ChunkRecord`。
    """

    parts: tuple[DocumentSection, ...]
    source_section_indexes: tuple[int, ...]


class DocumentSplitNode(BaseNode):
    """把 Markdown 一级切分为带标题层级关系的 section。

    标题层级示例：

    ```text
    # 第一章             parent_title = # 第一章
    ### 1.1 参数         parent_title = # 第一章（允许跳过 H2）
    #### 1.1.1 电压      parent_title = ### 1.1 参数
    ## 1.2 使用方法      parent_title = # 第一章
    #### 1.2.1 示例      parent_title = ## 1.2 使用方法
    ```

    遇到 ``## 1.2 使用方法`` 时会清空旧 H3/H4，因此最后一个 H4 不会错误地
    继承 ``### 1.1 参数`` 或 ``#### 1.1.1 电压``。
    """

    name = "document_split_node"

    _markdown_extensions = frozenset({".md", ".markdown"})
    _heading_pattern = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*$")
    _fence_pattern = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
    _image_line_pattern = re.compile(r"^\s*!\[[^\]]*]\([^\n]+\)\s*$")
    _link_line_pattern = re.compile(r"^\s*\[[^\]]+]\([^\n]+\)\s*$")
    _sentence_pattern = re.compile(r"[^。！？；.!?;]+[。！？；.!?;]?", re.S)

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """依次执行 Step 1-4，并把最终切片写入 ``state["chunks"]``。"""
        self.log_step("step_1", "获取并校验文档切分输入")
        inputs = self._validate_state(state)

        self.log_step("step_2", "按 Markdown 标题构建 section")
        sections = self._split_by_headings(inputs.md_content, inputs.file_title)
        self.logger.info(
            "Markdown 标题切分完成: section_count=%d",
            len(sections),
        )

        self.log_step("step_3", "切分超长 section 并合并相邻短切片")
        drafts = self._split_and_merge(
            sections,
            max_content_length=inputs.max_content_length,
            min_content_length=inputs.min_content_length,
            overlap_sentences=inputs.overlap_sentences,
        )
        self.logger.info(
            "长度治理完成: section_count=%d, draft_count=%d",
            len(sections),
            len(drafts),
        )

        self.log_step("step_4", "组装向量化节点所需的 ChunkRecord")
        chunks = self._assemble_chunks(
            drafts,
            file_title=inputs.file_title,
            source_path=inputs.md_path,
        )
        state["chunks"] = chunks
        self.logger.info(
            "最终切片组装完成: chunk_count=%d, total_chars=%d",
            len(chunks),
            sum(chunk["char_count"] for chunk in chunks),
        )
        return state

    def split_sections(self, state: ImportGraphState) -> list[DocumentSection]:
        """运行 Step 1/2 并返回 section，适合开发预览和单元测试。

        该方法不会读取文件内容，Markdown 正文必须已经由上游节点写入
        ``state["md_content"]``。调用后仅会规范化回写 ``file_title`` 和
        ``file_dir``，不会写入最终 ``chunks``。
        """
        self.log_step("step_1", "获取并校验文档切分输入")
        inputs = self._validate_state(state)

        self.log_step("step_2", "按 Markdown 标题构建 section")
        return self._split_by_headings(inputs.md_content, inputs.file_title)

    def _split_and_merge(
        self,
        sections: list[DocumentSection],
        *,
        max_content_length: int,
        min_content_length: int,
        overlap_sentences: int,
    ) -> list[ChunkDraft]:
        """执行 Step 3：先切分超长 section，再合并相邻短 draft。

        为什么顺序必须是“先长切、后短合”：

        1. 如果先合并，两个本来不长的 section 可能先变成超长文本，标题边界也随之
           模糊；先把所有内容约束到最大长度附近，后续合并更容易判断安全性。
        2. 短合并只是优化检索上下文，不是硬性要求。只要跨顶层章节，或合并后超过
           ``max_content_length``，宁可保留一个短 chunk 也不强行拼接。

        示例 A——超长正文：标题占 8 字、上限 100 字时，正文预算约为 90 字。
        LangChain 先按段落、换行、句末标点逐级寻找边界，实在找不到才按字符切分。

        示例 B——短 section：同属 ``# 安装`` 的 ``## 准备``（80 字）与
        ``## 接线``（120 字）在合并后不超过上限时可以合并，两个标题都会保留。

        示例 C——跨章禁止合并：``# 安装`` 的尾段即使很短，也不会与紧随其后的
        ``# 维护`` 合并，因为检索结果不能把两个顶层主题混成一个语义单元。
        """
        split_drafts: list[ChunkDraft] = []
        for section in sections:
            split_drafts.extend(
                self._split_long_section(
                    section,
                    max_content_length=max_content_length,
                    overlap_sentences=overlap_sentences,
                )
            )
        return self._merge_short_drafts(
            split_drafts,
            min_content_length=min_content_length,
            max_content_length=max_content_length,
        )

    def _split_long_section(
        self,
        section: DocumentSection,
        *,
        max_content_length: int,
        overlap_sentences: int,
    ) -> list[ChunkDraft]:
        """在每片重复标题上下文，并用 LangChain 切分超长正文。

        这里不把 ``title`` 截断为固定 50 字。标题本身是重要检索上下文，真正可用于
        正文的预算应是 ``max - len(title) - 两个换行``。如果标题本身已经占满上限，
        会抛出 ``DocumentSplitError``，提示调用方调整配置或治理异常标题。

        Markdown 特殊结构示例：

        - 围栏代码 `````python ... ````` 作为一个原子块，不会从代码中间切开；
        - 独占一行的 ``![接线图](a.png)`` 不会拆坏 Markdown 语法；
        - HTML 和管道表格统一由 ``MarkdownTableUtil`` 转成一维自然语言；HTML 中的
          跨行/跨列会先展开，因此节点本身不再包含任何表格解析分支；
        - 如果单个原子块自己就超过上限，只能完整保留并记录 warning。相比把代码或
          图片语法切坏，这是更可恢复的降级方式。
        """
        # 表格解析、类型判断和语义转译全部封装在 util，节点只消费普通文本结果。
        linearized_body = MarkdownTableUtil.process(section.body)
        if linearized_body != section.body:
            self.logger.debug(
                "文档表格已完成降维转译: source_section_index=%d",
                section.source_section_index,
            )
        normalized_section = replace(section, body=linearized_body)
        rendered = self._render_section(normalized_section)
        if len(rendered) <= max_content_length:
            return [self._draft_from_section(normalized_section)]

        title = normalized_section.title.strip()
        title_cost = len(title) + (2 if normalized_section.body.strip() else 0)
        body_budget = max_content_length - title_cost
        if body_budget <= 0:
            raise DocumentSplitError(
                message=(
                    "标题长度已占满切片上限，无法为正文分配空间: "
                    f"source_section_index={section.source_section_index}, "
                    f"title_length={len(title)}, max_content_length={max_content_length}"
                ),
                node_name=self.name,
            )

        body_parts = self._split_body_preserving_atoms(
            normalized_section.body,
            body_budget=body_budget,
            source_section_index=section.source_section_index,
        )
        body_parts = self._apply_sentence_overlap(
            body_parts,
            body_budget=body_budget,
            overlap_sentences=overlap_sentences,
        )

        self.logger.debug(
            "超长 section 已拆分: source_section_index=%d, part_count=%d, body_budget=%d",
            section.source_section_index,
            len(body_parts),
            body_budget,
        )
        return [
            self._draft_from_section(replace(normalized_section, body=body_part))
            for body_part in body_parts
        ]

    def _split_body_preserving_atoms(
        self,
        body: str,
        *,
        body_budget: int,
        source_section_index: int,
    ) -> list[str]:
        """组合自定义结构保护与 ``RecursiveCharacterTextSplitter``。

        LangChain 负责通用文本的递归边界选择；本方法只补充业务文档需要的原子结构
        保护。普通文本依次尝试 ``空行 → 换行 → 中文/英文标点 → 空格 → 字符``，
        因而既优先保持段落和句子，也保证没有自然边界的长串最终仍可切分。

        例如 ``说明段 + 代码围栏 + 后续段`` 会先分成三个 block。说明段和后续段交给
        LangChain；代码围栏整体作为一个 piece。最后按原顺序重新装箱，只要候选内容
        不超过 ``body_budget`` 就尽量放在同一片中。
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_budget,
            chunk_overlap=0,
            length_function=len,
            keep_separator=True,
            strip_whitespace=True,
            separators=[
                "\n\n",
                "\n",
                "。",
                "！",
                "？",
                "；",
                ".",
                "!",
                "?",
                ";",
                "，",
                ",",
                " ",
                "",
            ],
        )

        pieces: list[tuple[str, str, bool]] = []
        for block in self._partition_markdown_blocks(body):
            if block.atomic:
                pieces.append((block.text.strip("\n"), block.kind, True))
            else:
                pieces.extend(
                    (piece, block.kind, False)
                    for piece in splitter.split_text(block.text)
                )

        packed: list[str] = []
        current = ""
        for piece, kind, atomic in pieces:
            piece = piece.strip("\n")
            if not piece.strip():
                continue
            if atomic and len(piece) > body_budget:
                if current:
                    packed.append(current)
                    current = ""
                packed.append(piece)
                self.logger.warning(
                    "Markdown 原子块超过正文预算，保留完整结构: "
                    "source_section_index=%d, kind=%s, block_length=%d, body_budget=%d",
                    source_section_index,
                    kind,
                    len(piece),
                    body_budget,
                )
                continue

            candidate = piece if not current else f"{current}\n\n{piece}"
            if len(candidate) <= body_budget:
                current = candidate
            else:
                if current:
                    packed.append(current)
                current = piece
        if current:
            packed.append(current)
        return packed

    def _partition_markdown_blocks(self, body: str) -> list[_MarkdownBlock]:
        """识别必须整体保留的 Markdown 块，返回顺序不变的 block 列表。

        识别场景：完整或未闭合的代码围栏，以及独占一行的 Markdown 图片/链接。
        表格已经由 ``MarkdownTableUtil`` 转成普通自然语言；列表、引用和段落继续
        交给 LangChain 寻找更自然的切点。
        """
        lines = body.split("\n")
        blocks: list[_MarkdownBlock] = []
        plain_lines: list[str] = []

        def flush_plain() -> None:
            text = "\n".join(plain_lines).strip("\n")
            if text.strip():
                blocks.append(_MarkdownBlock(text=text, kind="text", atomic=False))
            plain_lines.clear()

        index = 0
        while index < len(lines):
            line = lines[index]
            opening_fence = self._match_opening_fence(line)
            if opening_fence is not None:
                flush_plain()
                fence_lines = [line]
                index += 1
                while index < len(lines):
                    fence_line = lines[index]
                    fence_lines.append(fence_line)
                    index += 1
                    if self._is_closing_fence(fence_line, opening_fence):
                        break
                blocks.append(
                    _MarkdownBlock(
                        text="\n".join(fence_lines),
                        kind="fenced_code",
                        atomic=True,
                    )
                )
                continue

            if self._image_line_pattern.match(line):
                flush_plain()
                blocks.append(_MarkdownBlock(text=line, kind="image", atomic=True))
                index += 1
                continue

            if self._link_line_pattern.match(line):
                flush_plain()
                blocks.append(_MarkdownBlock(text=line, kind="link", atomic=True))
                index += 1
                continue

            plain_lines.append(line)
            index += 1

        flush_plain()
        return blocks

    def _apply_sentence_overlap(
        self,
        body_parts: list[str],
        *,
        body_budget: int,
        overlap_sentences: int,
    ) -> list[str]:
        """把前片末尾若干完整句复制到后片开头，且不突破正文预算。

        示例：前片为“先断电。再开盖。”、后片为“取出电池。”，配置 overlap=1
        时后片会变成“再开盖。\n取出电池。”。若加入该句会超过上限，则逐句减少，
        最终允许零重叠；因此 overlap 是“尽力而为”的检索增强，不是破坏长度上限的
        硬指标。单个原子块本身已经超限时也不会继续追加重叠文本。
        """
        if overlap_sentences == 0 or len(body_parts) < 2:
            return body_parts

        overlapped = [body_parts[0]]
        for current in body_parts[1:]:
            if len(current) > body_budget:
                overlapped.append(current)
                continue

            previous_sentences = [
                sentence.strip()
                for sentence in self._sentence_pattern.findall(overlapped[-1])
                if sentence.strip()
            ]
            candidates = previous_sentences[-overlap_sentences:]
            while candidates:
                prefix = "".join(candidates)
                candidate = f"{prefix}\n{current}"
                if len(candidate) <= body_budget:
                    current = candidate
                    break
                candidates.pop(0)
            overlapped.append(current)
        return overlapped

    def _merge_short_drafts(
        self,
        drafts: list[ChunkDraft],
        *,
        min_content_length: int,
        max_content_length: int,
    ) -> list[ChunkDraft]:
        """短片先向前合并，失败后再尝试并入已经输出的前一片。

        “向前”能让标题后的短介绍和紧随其后的详细说明共同出现；文档尾部没有下一片
        时，再尝试“向后”合并可避免孤立尾巴。两个方向都必须满足：

        - 相邻且属于同一个顶层标题分支；
        - 合并后的 ``title + body`` 总长度不超过 ``max_content_length``。

        例如 ``# 安装 / ## 准备`` 与 ``# 安装 / ## 操作`` 可以合并，而
        ``# 安装 / ## 收尾`` 与下一章 ``# 维护`` 即使都很短也不会合并。
        """
        merged: list[ChunkDraft] = []
        index = 0
        while index < len(drafts):
            current = drafts[index]
            index += 1

            # 当前片仍偏短时，可连续吸收多个后继片；一旦语义或上限不满足便停止。
            while len(self._render_draft(current)) < min_content_length and index < len(drafts):
                following = drafts[index]
                if not self._can_merge(current, following, max_content_length):
                    break
                current = self._combine_drafts(current, following)
                index += 1

            # 文档尾部或前向合并受阻时，最后尝试并入已经落定的前一片。
            if (
                len(self._render_draft(current)) < min_content_length
                and merged
                and self._can_merge(merged[-1], current, max_content_length)
            ):
                merged[-1] = self._combine_drafts(merged[-1], current)
            else:
                merged.append(current)
        return merged

    def _can_merge(
        self,
        left: ChunkDraft,
        right: ChunkDraft,
        max_content_length: int,
    ) -> bool:
        """判断两个相邻 draft 是否同属顶层语义分支且合并后不过长。"""
        if self._top_level_key(left) != self._top_level_key(right):
            return False
        return len(self._render_draft(self._combine_drafts(left, right))) <= max_content_length

    @staticmethod
    def _combine_drafts(left: ChunkDraft, right: ChunkDraft) -> ChunkDraft:
        """按原文顺序合并，并稳定去重原始 section 索引。"""
        indexes = tuple(dict.fromkeys(left.source_section_indexes + right.source_section_indexes))
        return ChunkDraft(
            parts=left.parts + right.parts,
            source_section_indexes=indexes,
        )

    def _assemble_chunks(
        self,
        drafts: list[ChunkDraft],
        *,
        file_title: str,
        source_path: Path,
    ) -> list[ChunkRecord]:
        """执行 Step 4：将内部 draft 转成稳定、可序列化的 ChunkRecord。

        单 section 示例：``source_titles=["## 电压"]``，``heading_path`` 保留完整
        ``["# 测量", "## 电压"]``。多个短 section 合并后，``content`` 会依次保留
        每个标题和正文，``source_titles`` 也记录全部标题；``heading_path`` 则取所有
        part 的最长公共前缀，例如两个 H2 兄弟合并后得到 ``["# 测量"]``。

        ``content`` 是 embedding 的主字段，``body`` 是不含标题的正文汇总；
        ``char_count`` 始终等于 ``len(content)``，便于 Step 5 做统计和故障排查。
        """
        chunks: list[ChunkRecord] = []
        for chunk_index, draft in enumerate(drafts):
            first_part = draft.parts[0]
            content = self._render_draft(draft)
            chunks.append(
                ChunkRecord(
                    chunk_index=chunk_index,
                    file_title=file_title,
                    title=first_part.title,
                    parent_title=first_part.parent_title,
                    heading_path=list(self._common_heading_path(draft.parts)),
                    source_titles=list(dict.fromkeys(part.title for part in draft.parts)),
                    source_section_indexes=list(draft.source_section_indexes),
                    body="\n\n".join(
                        part.body.strip("\n")
                        for part in draft.parts
                        if part.body.strip()
                    ),
                    content=content,
                    char_count=len(content),
                    source_path=str(source_path),
                )
            )
        return chunks

    @staticmethod
    def _draft_from_section(section: DocumentSection) -> ChunkDraft:
        """把一个 section 或其长文片段包装成 Step 3 draft。"""
        return ChunkDraft(
            parts=(section,),
            source_section_indexes=(section.source_section_index,),
        )

    @staticmethod
    def _render_section(section: DocumentSection) -> str:
        """按 ``title + 空行 + body`` 生成真正参与长度计算和 embedding 的文本。"""
        title = section.title.strip()
        body = section.body.strip("\n")
        if title and body:
            return f"{title}\n\n{body}"
        return title or body

    @classmethod
    def _render_draft(cls, draft: ChunkDraft) -> str:
        """合并时保留每个来源标题，避免正文脱离原始小节语义。"""
        return "\n\n".join(
            rendered
            for part in draft.parts
            if (rendered := cls._render_section(part))
        )

    @staticmethod
    def _top_level_key(draft: ChunkDraft) -> str:
        """返回 draft 所属最高层有效标题，用作短合并的语义边界。"""
        first_path = draft.parts[0].heading_path
        return first_path[0] if first_path else draft.parts[0].title

    @staticmethod
    def _common_heading_path(parts: tuple[DocumentSection, ...]) -> tuple[str, ...]:
        """计算合并后各 part 的最长公共标题路径。"""
        if not parts:
            return ()
        common: list[str] = []
        for index, title in enumerate(parts[0].heading_path):
            # 路径必须从根部连续相同；中间一层不同后，更深层即使同名也没有共同语义。
            if all(
                len(part.heading_path) > index and part.heading_path[index] == title
                for part in parts[1:]
            ):
                common.append(title)
            else:
                break
        return tuple(common)

    def _validate_state(self, state: ImportGraphState) -> SplitInputs:
        """校验状态和切分配置，并统一不同操作系统的换行符。"""
        if not isinstance(state, dict):
            raise StateFieldError(
                node_name=self.name,
                field_name="state",
                expected_type=dict,
            )

        # 不能用 ``if not md_content``：空字符串代表合法的空文档，应返回零个 section。
        md_content = state.get("md_content")
        if not isinstance(md_content, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="md_content",
                expected_type=str,
            )

        # removeprefix 只移除文档开头的 BOM，不会误删正文里合法的同类字符。
        normalized_content = (
            md_content.replace("\r\n", "\n")
            .replace("\r", "\n")
            .removeprefix("\ufeff")
        )

        md_path = self._require_markdown_path(state.get("md_path"))
        file_title = self._resolve_file_title(state, md_path)
        file_dir = self._resolve_file_dir(state.get("file_dir"), md_path)
        self._validate_split_config()

        # 统一回写一次，后续 Step 3-5 以及下游节点无需重复处理兜底规则。
        state["file_title"] = file_title
        state["file_dir"] = str(file_dir)

        return SplitInputs(
            md_content=normalized_content,
            file_title=file_title,
            md_path=md_path,
            file_dir=file_dir,
            max_content_length=self.config.max_content_length,
            min_content_length=self.config.min_content_length,
            overlap_sentences=self.config.overlap_sentences,
        )

    def _require_markdown_path(self, raw_path: object) -> Path:
        """校验并返回规范化 Markdown 路径；文件存在性已由上游节点负责。"""
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise StateFieldError(
                node_name=self.name,
                field_name="md_path",
                expected_type=str,
            )

        md_path = Path(raw_path.strip()).expanduser()
        if md_path.suffix.casefold() not in self._markdown_extensions:
            raise StateFieldError(
                node_name=self.name,
                field_name="md_path",
                expected_type=str,
                message=f"状态字段 'md_path' 必须指向 Markdown 文件: {md_path}",
            )
        return md_path.resolve()

    def _resolve_file_title(
        self,
        state: ImportGraphState,
        md_path: Path,
    ) -> str:
        """按显式标题、原始导入路径、处理后路径的优先级确定文件标题。"""
        raw_title = state.get("file_title")
        if raw_title is not None and not isinstance(raw_title, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="file_title",
                expected_type=str,
            )
        if isinstance(raw_title, str) and raw_title.strip():
            return raw_title.strip()

        import_file_path = state.get("import_file_path")
        if import_file_path is not None and not isinstance(import_file_path, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="import_file_path",
                expected_type=str,
            )
        if isinstance(import_file_path, str) and import_file_path.strip():
            import_title = Path(import_file_path.strip()).stem.strip()
            if import_title:
                return import_title

        fallback_title = md_path.stem.strip()
        if fallback_title:
            return fallback_title
        raise StateFieldError(
            node_name=self.name,
            field_name="file_title",
            expected_type=str,
            message="无法从 file_title、import_file_path 或 md_path 确定文件标题",
        )

    def _resolve_file_dir(self, raw_dir: object, md_path: Path) -> Path:
        """优先使用显式输出目录，否则回退到 Markdown 所在目录。"""
        if raw_dir is not None and not isinstance(raw_dir, str):
            raise StateFieldError(
                node_name=self.name,
                field_name="file_dir",
                expected_type=str,
            )
        if isinstance(raw_dir, str) and raw_dir.strip():
            return Path(raw_dir.strip()).expanduser().resolve()
        return md_path.parent

    def _validate_split_config(self) -> None:
        """校验 Step 3 会使用的长度参数，尽早阻止无效配置进入流程。"""
        max_length = self.config.max_content_length
        min_length = self.config.min_content_length
        overlap = self.config.overlap_sentences

        # bool 是 int 的子类，显式排除可避免 True 被误当成长度 1。
        if (
            not self._is_integer(max_length)
            or not self._is_integer(min_length)
            or max_length <= 0
            or min_length <= 0
            or max_length <= min_length
        ):
            raise ConfigurationError(
                message=(
                    "文档切分长度配置无效: max_content_length 必须大于 "
                    "min_content_length，且两者都必须是正整数"
                ),
                node_name=self.name,
            )
        if not self._is_integer(overlap) or overlap < 0:
            raise ConfigurationError(
                message="overlap_sentences 必须是大于等于 0 的整数",
                node_name=self.name,
            )

    def _split_by_headings(
        self,
        md_content: str,
        file_title: str,
    ) -> list[DocumentSection]:
        """按 ATX H1-H6 标题切分，并为每个 section 找到最近父标题。

        Args:
            md_content: 已统一为 ``\n`` 换行的完整 Markdown 内容。
            file_title: 无显式标题时使用的文档兜底标题。

        Returns:
            按原文顺序排列的 section。每个 section 都保留当前标题、最近父标题、
            完整有效标题路径、正文和稳定序号，供 Step 3 切分/合并继续使用。

        场景 1：正常的连续标题层级

        ```text
        # 第一章
        ## 基础知识
        ### 安全说明
        ```

        输出关系：

        ```text
        # 第一章      → parent=# 第一章，path=(# 第一章)
        ## 基础知识   → parent=# 第一章，path=(# 第一章, ## 基础知识)
        ### 安全说明  → parent=## 基础知识，path=(# 第一章, ## 基础知识, ### 安全说明)
        ```

        H1 没有更高层祖先，所以按照当前业务约定以自身作为父标题。

        场景 2：标题跳级，向上跳过空层级寻找父标题

        ```text
        # 第一章
        ### 电气参数
        ```

        此时 ``hierarchy[2]`` 为空，H3 会继续向上检查 ``hierarchy[1]``，
        最终得到 ``parent=# 第一章``，路径为 ``(# 第一章, ### 电气参数)``。
        后续 Step 3 即使拆分“电气参数”的正文，也能把 H1 语义一起带入 chunk。

        场景 3：回到同级或上级，必须清除已经失效的下级标题

        ```text
        # 第一章
        ## 1.1 测量
        ### 1.1.1 电压
        ## 1.2 维护
        #### 1.2.1 更换电池
        ```

        读到 ``## 1.2 维护`` 时会清空旧 H3-H6。最后一个 H4 向上查找时，
        H3 已为空，因此得到 ``parent=## 1.2 维护``，而不会错误继承
        ``### 1.1.1 电压``。其路径为
        ``(# 第一章, ## 1.2 维护, #### 1.2.1 更换电池)``。

        场景 4：第一个标题前存在前言

        ```text
        本文介绍设备的基本用法。

        # 第一章
        正文
        ```

        前言独立形成 level=0 的 section，``title``、``parent_title`` 和
        ``heading_path`` 均使用 ``file_title``；后面的 H1 正常形成新 section。

        场景 5：全文没有标题或只有普通正文

        ```text
        第一段正文。
        第二段正文。
        ```

        全文形成一个 level=0 的 section，标题使用 ``file_title``。这样没有规范
        Markdown 标题的旧文档仍能进入 Step 3，而不会丢失整篇内容。

        场景 6：文档从孤立的子标题开始

        ```text
        ### 参数说明
        正文
        ```

        因为 H1/H2 都不存在，H3 找不到祖先，最终以自身作为父标题，路径为
        ``(### 参数说明)``。这是容错策略，不会虚构一个不存在的 H1/H2。

        场景 7：连续标题，中间没有正文

        ```text
        # 第一章
        ## 1.1 概述
        正文
        ```

        遇到 H2 时仍会 flush H1，因此 H1 section 的 ``body=""``。保留空正文
        标题能让 Step 3 决定是否与后续短 section 合并，而不是在 Step 2 提前丢弃结构。

        场景 8：代码围栏中的井号不是标题

        ```text
        # 示例
        ````python
        ## 这是代码内容
        ```
        ### 三个反引号不足以关闭四个反引号围栏
        ````
        ## 真正的下一个标题
        ```

        ``## 这是代码内容`` 和 ``### 三个反引号...`` 都保留在“# 示例”的 body。
        关闭行必须与打开行使用相同字符，并且长度不少于打开围栏；``~~~`` 也不能
        关闭反引号围栏。未闭合围栏会把文档剩余内容全部视为围栏正文。

        场景 9：看起来像标题、但不符合 ATX 规则的行

        ```text
        #没有空格
        ####### 七级标题
            ## 四空格缩进
           ### 三空格缩进的合法标题
        ```

        前三行作为普通正文；最后一行允许 0-3 个前导空格且是 H3，因此正常切分。
        本期只支持 ATX H1-H6，不识别 Setext 标题或 HTML 标题。

        场景 10：空内容与纯空白内容

        ``md_content=""`` 或只包含空格、换行时返回空列表，不创建没有实际内容的
        file_title section。空文档仍是合法输入，后续 Step 5 可以记录零切片统计。

        为什么这些字段能支撑 Step 3：

        - ``title`` 与 ``body`` 分开，长正文切分时可以为每段重复标题上下文；
        - ``heading_path`` 保留跨级祖先，切分后不会失去所属章节；
        - ``parent_title`` 便于短内容判断是否能在同一语义分支合并；
        - ``source_section_index`` 保证切分、合并后仍可恢复原始顺序；
        - fenced code block 完整保留在 body，Step 3 可以把它视为 Markdown 原子结构。
        """
        # hierarchy 像一条“当前标题路径缓存”：每个槽位只保存最近的同级标题。
        # 例如 H1→H3 时索引 2 为空；H3 找父标题时会跳过它并继续找到 H1。
        sections: list[DocumentSection] = []
        hierarchy = [""] * 7  # 索引 1-6 对应 H1-H6，索引 0 不使用。

        # 下面四个变量描述“尚未写入 sections 的当前章节”。遇到新标题或文档结束时，
        # flush_current_section 会把它们一次性封装，避免逐行构造大量临时字符串。
        body_lines: list[str] = []
        current_title = ""
        current_level = 0
        active_fence: _FenceMarker | None = None

        def flush_current_section() -> None:
            """把当前标题和累计正文封装为 section，不修改标题层级缓存。"""
            # list 中逐行积累、最后 join 是 Python 常用写法，避免循环中反复拼接大字符串。
            body = "\n".join(body_lines).strip("\n")

            # 没有标题且正文也只有空白时不生成 section；但“有标题、空正文”必须保留。
            if not current_title and not body.strip():
                return

            section_title = current_title or file_title
            if current_level == 0:
                # 首标题之前的前言没有 Markdown 层级，只能归到文件标题。
                parent_title = file_title
                heading_path = (file_title,)
            else:
                parent_title = ""
                # 例：current_level=4 时依次检查 H3、H2、H1，找到第一个有效标题即停止。
                # 这种倒序查找同时支持完整层级和 H1→H4 之类的跳级文档。
                for level in range(current_level - 1, 0, -1):
                    if hierarchy[level]:
                        parent_title = hierarchy[level]
                        break
                # H1 或没有任何祖先的孤立子标题，以自身作为父标题。
                parent_title = parent_title or current_title

                # 切片只保留非空槽位：H1→H3 会得到 (H1, H3)，不会塞入虚假的 H2。
                heading_path = tuple(
                    title
                    for title in hierarchy[1 : current_level + 1]
                    if title
                )

            # len(sections) 正好是下一个连续索引，不需要额外维护可变计数器。
            sections.append(
                DocumentSection(
                    title=section_title,
                    heading_level=current_level,
                    parent_title=parent_title,
                    heading_path=heading_path,
                    body=body,
                    source_section_index=len(sections),
                )
            )

        # 每行严格按以下优先级处理：
        # 1. 已在代码围栏内；2. 新围栏起点；3. 普通正文；4. 合法标题。
        # 围栏判断必须早于标题判断，否则代码块里的“# 注释”会被错误切分。
        for content_line in md_content.split("\n"):
            if active_fence is not None:
                # 围栏内所有行（包括关闭行本身）都属于当前 section 的正文。
                body_lines.append(content_line)
                if self._is_closing_fence(content_line, active_fence):
                    active_fence = None
                continue

            opening_fence = self._match_opening_fence(content_line)
            if opening_fence is not None:
                # 记录字符和长度，后面只有匹配的关闭围栏才能退出代码块模式。
                active_fence = opening_fence
                body_lines.append(content_line)
                continue

            heading_match = self._heading_pattern.match(content_line)
            if heading_match is None or not heading_match.group(2).strip():
                # 不符合 ATX 标题规则的行原样保留，包括图片、表格、列表和普通段落。
                body_lines.append(content_line)
                continue

            # 新标题属于下一个 section，因此顺序必须是：保存旧章节 → 更新新标题状态。
            # 如果反过来，旧正文就会被错误地挂到新标题下面。
            flush_current_section()
            body_lines = []

            current_level = len(heading_match.group(1))
            current_title = content_line
            hierarchy[current_level] = current_title

            # 例：H1→H2(A)→H3→H2(B)。写入 H2(B) 后必须清空 H3-H6，
            # 否则 B 下方新出现的 H4 会错误继承 A 的旧 H3。
            for lower_level in range(current_level + 1, 7):
                hierarchy[lower_level] = ""

        # 循环只会在“遇到下一个标题”时保存上一节，最后一节需要在 EOF 主动保存。
        flush_current_section()
        return sections

    @classmethod
    def _match_opening_fence(cls, line: str) -> _FenceMarker | None:
        """识别代码围栏起点，同时遵循反引号 info string 的基本限制。"""
        match = cls._fence_pattern.match(line)
        if match is None:
            return None

        marker_text = match.group(1)
        info_string = match.group(2)
        # CommonMark 不允许反引号围栏的 info string 再包含反引号。
        if marker_text[0] == "`" and "`" in info_string:
            return None
        return _FenceMarker(character=marker_text[0], length=len(marker_text))

    @classmethod
    def _is_closing_fence(cls, line: str, active_fence: _FenceMarker) -> bool:
        """只有同字符且长度足够的纯围栏行，才能关闭当前代码块。"""
        match = cls._fence_pattern.match(line)
        if match is None or match.group(2).strip():
            return False

        marker_text = match.group(1)
        return (
            marker_text[0] == active_fence.character
            and len(marker_text) >= active_fence.length
        )

    @staticmethod
    def _is_integer(value: object) -> bool:
        """判断配置是否为真正整数，并排除 Python 中属于 int 子类的 bool。"""
        return isinstance(value, int) and not isinstance(value, bool)


def main() -> None:
    """使用小长度配置直观展示 Step 2 section 与 Step 3/4 chunks。"""
    setup_logging()
    example_markdown = """文档前言，不属于任何显式标题。

# 第一章
章节说明。

### 1.1 参数
H1 直接跳到 H3，父标题仍应是“# 第一章”。参数正文会重复多次，
用于演示 LangChain 如何在句子边界附近拆分长 section。安全测量。正确接线。
检查量程。读取数值。安全测量。正确接线。检查量程。读取数值。

```python
# 这是代码注释，不是标题
```

## 1.2 使用方法
回到 H2 后，旧 H3 已失效。

#### 1.2.1 示例
该 H4 的父标题应是“## 1.2 使用方法”。
"""
    state: ImportGraphState = {
        "md_content": example_markdown,
        "md_path": "document_split_example.md",
        "file_title": "文档切分示例",
    }

    node = DocumentSplitNode(
        config=ImportConfig(
            max_content_length=120,
            min_content_length=45,
            overlap_sentences=1,
        )
    )
    sections = node.split_sections(state.copy())
    print("\n=== Step 2: sections ===")
    for section in sections:
        print(
            f"[{section.source_section_index}] level={section.heading_level} "
            f"title={section.title!r} parent={section.parent_title!r} "
            f"path={' > '.join(section.heading_path)}"
        )

    result = node.process(state)
    print("\n=== Step 3/4: final chunks ===")
    for chunk in result["chunks"]:
        print(
            f"[{chunk['chunk_index']}] chars={chunk['char_count']} "
            f"sources={chunk['source_section_indexes']} "
            f"path={' > '.join(chunk['heading_path'])}"
        )
        print(chunk["content"])
        print("---")


if __name__ == "__main__":
    main()
