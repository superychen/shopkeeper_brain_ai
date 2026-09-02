"""DocumentSplitNode Step 1/2 单元测试。"""

import unittest
from pathlib import Path

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    DocumentSplitError,
    StateFieldError,
)
from knowledge.processor.import_processor.nodes.document_split_node import (
    ChunkDraft,
    DocumentSection,
    DocumentSplitNode,
)
from knowledge.processor.import_processor.utils.markdown_table_util import (
    MarkdownTableUtil,
)


class DocumentSplitNodeTest(unittest.TestCase):
    """覆盖输入校验、标题层级、父标题和代码围栏场景。"""

    def setUp(self) -> None:
        # Step 1 只校验和规范化路径，不读取文件；使用仓库内逻辑路径避免测试产生文件。
        self.workspace = Path("knowledge/test/fixtures").resolve()
        self.node = self._make_node()

    @staticmethod
    def _make_node(
        *,
        max_length: object = 2000,
        min_length: object = 500,
        overlap: object = 1,
    ) -> DocumentSplitNode:
        config = ImportConfig(
            max_content_length=max_length,
            min_content_length=min_length,
            overlap_sentences=overlap,
        )
        return DocumentSplitNode(config=config)

    def _state(self, md_content: str, **overrides: object) -> dict:
        state = {
            "md_content": md_content,
            "md_path": str(self.workspace / "manual_new.md"),
            "import_file_path": str(self.workspace / "manual.pdf"),
        }
        state.update(overrides)
        return state

    @staticmethod
    def _section(
        title: str,
        body: str,
        index: int,
        *,
        path: tuple[str, ...] | None = None,
    ) -> DocumentSection:
        """构造 Step 3 测试 section，减少与测试意图无关的样板字段。"""
        heading_path = path or (title,)
        return DocumentSection(
            title=title,
            heading_level=len(title) - len(title.lstrip("#")),
            parent_title=heading_path[-2] if len(heading_path) > 1 else title,
            heading_path=heading_path,
            body=body,
            source_section_index=index,
        )

    def test_validate_state_normalizes_input_and_derives_metadata(self) -> None:
        state = self._state("\ufeff第一行\r\n第二行\r第三行")

        inputs = self.node._validate_state(state)

        self.assertEqual(inputs.md_content, "第一行\n第二行\n第三行")
        self.assertEqual(inputs.file_title, "manual")
        self.assertEqual(inputs.md_path, (self.workspace / "manual_new.md").resolve())
        self.assertEqual(inputs.file_dir, self.workspace.resolve())
        self.assertEqual(state["file_title"], "manual")
        self.assertEqual(state["file_dir"], str(self.workspace.resolve()))

    def test_file_title_uses_explicit_import_and_md_path_priority(self) -> None:
        scenarios = [
            ({"file_title": " 显式标题 "}, "显式标题"),
            ({"file_title": "", "import_file_path": "source.PDF"}, "source"),
            ({"file_title": "", "import_file_path": ""}, "manual_new"),
        ]

        for overrides, expected_title in scenarios:
            with self.subTest(expected_title=expected_title):
                inputs = self.node._validate_state(self._state("正文", **overrides))
                self.assertEqual(inputs.file_title, expected_title)

    def test_empty_document_is_valid_but_missing_content_is_not(self) -> None:
        self.assertEqual(self.node.split_sections(self._state("")), [])

        with self.assertRaises(StateFieldError):
            self.node._validate_state(
                {"md_path": str(self.workspace / "manual.md")}
            )

        with self.assertRaises(StateFieldError):
            self.node._validate_state(self._state(None))

    def test_invalid_markdown_path_and_metadata_types_fail(self) -> None:
        invalid_states = [
            self._state("正文", md_path="manual.txt"),
            self._state("正文", file_title=123),
            self._state("正文", file_title="", import_file_path=123),
            self._state("正文", file_dir=123),
        ]

        for state in invalid_states:
            with self.subTest(state=state), self.assertRaises(StateFieldError):
                self.node._validate_state(state)

    def test_invalid_length_config_fails_early(self) -> None:
        invalid_configs = [
            {"max_length": 0},
            {"min_length": 0},
            {"max_length": 500, "min_length": 500},
            {"max_length": 100, "min_length": 500},
            {"max_length": True},
            {"overlap": -1},
            {"overlap": False},
        ]

        for config_values in invalid_configs:
            with self.subTest(config_values=config_values):
                node = self._make_node(**config_values)
                with self.assertRaises(ConfigurationError):
                    node._validate_state(self._state("正文"))

    def test_parent_lookup_handles_skipped_and_reset_levels(self) -> None:
        markdown = """# 第一章
章节说明
### 1.1 参数
参数正文
#### 1.1.1 电压
电压正文
## 1.2 使用方法
使用正文
#### 1.2.1 示例
示例正文
# 第二章
第二章正文"""

        sections = self.node._split_by_headings(markdown, "测试文档")

        self.assertEqual(
            [section.parent_title for section in sections],
            [
                "# 第一章",
                "# 第一章",
                "### 1.1 参数",
                "# 第一章",
                "## 1.2 使用方法",
                "# 第二章",
            ],
        )
        self.assertEqual(
            sections[4].heading_path,
            ("# 第一章", "## 1.2 使用方法", "#### 1.2.1 示例"),
        )
        self.assertNotIn("### 1.1 参数", sections[4].heading_path)

    def test_all_heading_levels_build_complete_path(self) -> None:
        markdown = "\n".join(
            [
                "# H1",
                "## H2",
                "### H3",
                "#### H4",
                "##### H5",
                "###### H6",
            ]
        )

        sections = self.node._split_by_headings(markdown, "标题层级")

        self.assertEqual([section.heading_level for section in sections], list(range(1, 7)))
        self.assertEqual(sections[-1].parent_title, "##### H5")
        self.assertEqual(
            sections[-1].heading_path,
            ("# H1", "## H2", "### H3", "#### H4", "##### H5", "###### H6"),
        )

    def test_preamble_no_heading_and_isolated_child_have_fallbacks(self) -> None:
        preamble_sections = self.node._split_by_headings(
            "前言\n\n# 正文\n内容",
            "文件标题",
        )
        self.assertEqual(preamble_sections[0].title, "文件标题")
        self.assertEqual(preamble_sections[0].heading_level, 0)
        self.assertEqual(preamble_sections[0].parent_title, "文件标题")
        self.assertEqual(preamble_sections[0].heading_path, ("文件标题",))

        no_heading = self.node._split_by_headings("只有正文", "文件标题")
        self.assertEqual(len(no_heading), 1)
        self.assertEqual(no_heading[0].title, "文件标题")

        isolated_child = self.node._split_by_headings("### 孤立 H3\n内容", "文件标题")
        self.assertEqual(isolated_child[0].parent_title, "### 孤立 H3")
        self.assertEqual(isolated_child[0].heading_path, ("### 孤立 H3",))

    def test_invalid_heading_forms_remain_in_body(self) -> None:
        markdown = """#没有空格
####### 七级标题
    ## 四空格缩进
   ### 合法标题
正文"""

        sections = self.node._split_by_headings(markdown, "文件标题")

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].title, "文件标题")
        self.assertIn("#没有空格", sections[0].body)
        self.assertIn("####### 七级标题", sections[0].body)
        self.assertIn("    ## 四空格缩进", sections[0].body)
        self.assertEqual(sections[1].title, "   ### 合法标题")
        self.assertEqual(sections[1].heading_level, 3)

    def test_fenced_code_ignores_headings_until_matching_close(self) -> None:
        markdown = """# 正常标题
````python
## 围栏内标题
```
### 短围栏不能关闭
~~~~
#### 不同字符不能关闭
````
## 第二个标题
~~~text
# 波浪围栏内标题
~~~
### 第二个标题的子标题
正文"""

        sections = self.node._split_by_headings(markdown, "代码示例")

        self.assertEqual(
            [section.title for section in sections],
            ["# 正常标题", "## 第二个标题", "### 第二个标题的子标题"],
        )
        self.assertIn("## 围栏内标题", sections[0].body)
        self.assertIn("### 短围栏不能关闭", sections[0].body)
        self.assertIn("#### 不同字符不能关闭", sections[0].body)
        self.assertIn("# 波浪围栏内标题", sections[1].body)

    def test_unclosed_fence_keeps_remaining_headings_in_body(self) -> None:
        sections = self.node._split_by_headings(
            "# 标题\n```python\n## 仍是代码\n### 也不是标题",
            "代码示例",
        )

        self.assertEqual(len(sections), 1)
        self.assertIn("## 仍是代码", sections[0].body)
        self.assertIn("### 也不是标题", sections[0].body)

    def test_consecutive_headings_and_whitespace_document(self) -> None:
        sections = self.node._split_by_headings("# H1\n## H2\n正文", "文件标题")
        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].body, "")
        self.assertEqual(sections[1].parent_title, "# H1")
        self.assertEqual([section.source_section_index for section in sections], [0, 1])
        self.assertEqual(self.node._split_by_headings(" \n\n  ", "文件标题"), [])

    def test_process_publishes_final_chunks_and_replaces_stale_value(self) -> None:
        state = self._state("# 标题\n正文", chunks=["existing-result"])

        result = self.node.process(state)

        self.assertIs(result, state)
        self.assertEqual(len(result["chunks"]), 1)
        self.assertEqual(result["chunks"][0]["content"], "# 标题\n\n正文")
        self.assertEqual(result["chunks"][0]["char_count"], len("# 标题\n\n正文"))
        self.assertEqual(result["chunks"][0]["source_section_indexes"], [0])

    def test_process_empty_document_publishes_empty_chunks(self) -> None:
        state = self._state("", chunks=["stale"])

        result = self.node.process(state)

        self.assertEqual(result["chunks"], [])

    def test_long_section_uses_title_budget_and_respects_max_length(self) -> None:
        node = self._make_node(max_length=60, min_length=10, overlap=0)
        markdown = "# 参数\n" + "第一句说明。第二句说明。第三句说明。" * 8

        chunks = node.process(self._state(markdown))["chunks"]

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["content"].startswith("# 参数\n\n") for chunk in chunks))
        self.assertTrue(all(chunk["char_count"] <= 60 for chunk in chunks))
        self.assertTrue(all(chunk["source_section_indexes"] == [0] for chunk in chunks))

    def test_title_that_consumes_max_length_raises_clear_error(self) -> None:
        node = self._make_node(max_length=20, min_length=5, overlap=0)
        markdown = f"# {'长' * 20}\n正文"

        with self.assertRaises(DocumentSplitError):
            node.process(self._state(markdown))

    def test_sentence_overlap_is_best_effort_and_never_exceeds_budget(self) -> None:
        node = self._make_node(max_length=100, min_length=10, overlap=1)

        result = node._apply_sentence_overlap(
            ["先断电。再开盖。", "取出电池。", "重新装好。"],
            body_budget=14,
            overlap_sentences=1,
        )

        self.assertEqual(result[1], "再开盖。\n取出电池。")
        self.assertEqual(result[2], "取出电池。\n重新装好。")
        self.assertTrue(all(len(part) <= 14 for part in result))

    def test_fenced_code_is_not_split_in_the_middle(self) -> None:
        node = self._make_node(max_length=70, min_length=10, overlap=0)
        code = "```python\nvalue = 1\n# 代码里的标题\nprint(value)\n```"
        markdown = f"# 示例\n{'前置说明。' * 10}\n\n{code}\n\n{'后续说明。' * 10}"

        chunks = node.process(self._state(markdown))["chunks"]

        chunks_with_code = [chunk for chunk in chunks if "```python" in chunk["content"]]
        self.assertEqual(len(chunks_with_code), 1)
        self.assertIn(code, chunks_with_code[0]["content"])
        self.assertEqual(sum(chunk["content"].count("```") for chunk in chunks), 2)

    def test_standalone_image_and_link_are_detected_as_atomic_blocks(self) -> None:
        blocks = self.node._partition_markdown_blocks(
            "普通说明\n\n![接线图](images/wiring.png)\n[完整手册](manual.pdf)"
        )

        self.assertEqual([block.kind for block in blocks], ["text", "image", "link"])
        self.assertEqual([block.atomic for block in blocks], [False, True, True])

    def test_oversized_atomic_block_is_preserved_with_warning(self) -> None:
        node = self._make_node(max_length=45, min_length=10, overlap=0)
        code = "```text\n" + ("x" * 70) + "\n```"

        with self.assertLogs(node.logger, level="WARNING") as captured:
            chunks = node.process(self._state(f"# 示例\n{code}"))["chunks"]

        self.assertIn(code, chunks[0]["content"])
        self.assertGreater(chunks[0]["char_count"], 45)
        self.assertTrue(any("原子块超过正文预算" in line for line in captured.output))

    def test_html_table_linearizer_repeats_rowspan_context(self) -> None:
        html_table = (
            '<table><tr><th>型号</th><th>量程</th></tr>'
            '<tr><td rowspan="2">A1</td><td>20V</td></tr>'
            '<tr><td>200V</td></tr></table>'
        )

        result = MarkdownTableUtil.process(html_table)

        self.assertEqual(
            result,
            "【表格转译】\n- 【型号:A1，量程:20V】\n"
            "- 【型号:A1，量程:200V】\n"
            "【表格转译结束】",
        )

    def test_long_html_table_is_linearized_before_splitting(self) -> None:
        node = self._make_node(max_length=55, min_length=10, overlap=0)
        rows = "".join(f"<tr><td>A{i}</td><td>{i * 20}V</td></tr>" for i in range(8))
        markdown = f"# 参数\n<table>{rows}</table>"

        chunks = node.process(self._state(markdown))["chunks"]
        full_content = "\n".join(chunk["content"] for chunk in chunks)

        self.assertNotIn("<table>", full_content)
        self.assertIn("【表格转译】", full_content)
        self.assertIn("- A7：140V", full_content)

    def test_short_draft_merges_forward_within_same_top_level_branch(self) -> None:
        node = self._make_node(max_length=100, min_length=45, overlap=0)
        sections = [
            self._section("## 准备", "短介绍", 0, path=("# 安装", "## 准备")),
            self._section("## 接线", "详细步骤" * 5, 1, path=("# 安装", "## 接线")),
        ]

        drafts = node._split_and_merge(
            sections,
            max_content_length=100,
            min_content_length=45,
            overlap_sentences=0,
        )

        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].source_section_indexes, (0, 1))
        self.assertIn("## 准备", node._render_draft(drafts[0]))
        self.assertIn("## 接线", node._render_draft(drafts[0]))

    def test_short_tail_merges_backward_when_no_following_draft_exists(self) -> None:
        node = self._make_node(max_length=100, min_length=35, overlap=0)
        first = self._section("## 操作", "足够长的操作说明" * 4, 0, path=("# 安装", "## 操作"))
        tail = self._section("## 注意", "短尾巴", 1, path=("# 安装", "## 注意"))
        drafts = [node._draft_from_section(first), node._draft_from_section(tail)]

        merged = node._merge_short_drafts(
            drafts,
            min_content_length=35,
            max_content_length=100,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_section_indexes, (0, 1))

    def test_short_drafts_do_not_merge_across_top_level_headings(self) -> None:
        node = self._make_node(max_length=100, min_length=40, overlap=0)
        sections = [
            self._section("## 收尾", "短", 0, path=("# 安装", "## 收尾")),
            self._section("# 维护", "也很短", 1, path=("# 维护",)),
        ]

        drafts = node._split_and_merge(
            sections,
            max_content_length=100,
            min_content_length=40,
            overlap_sentences=0,
        )

        self.assertEqual(len(drafts), 2)

    def test_step4_uses_common_heading_path_and_preserves_all_titles(self) -> None:
        node = self._make_node(max_length=100, min_length=50, overlap=0)
        parts = (
            self._section("## 准备", "工具", 2, path=("# 安装", "## 准备")),
            self._section("## 接线", "步骤", 3, path=("# 安装", "## 接线")),
        )
        draft = ChunkDraft(parts=parts, source_section_indexes=(2, 3))

        chunks = node._assemble_chunks(
            [draft],
            file_title="设备手册",
            source_path=Path("manual.md").resolve(),
        )

        self.assertEqual(chunks[0]["heading_path"], ["# 安装"])
        self.assertEqual(chunks[0]["source_titles"], ["## 准备", "## 接线"])
        self.assertEqual(chunks[0]["source_section_indexes"], [2, 3])
        self.assertIn("## 准备\n\n工具", chunks[0]["content"])
        self.assertIn("## 接线\n\n步骤", chunks[0]["content"])

    def test_base_node_preserves_classified_validation_error(self) -> None:
        with self.assertRaises(StateFieldError):
            self.node({"md_path": str(self.workspace / "manual.md")})


if __name__ == "__main__":
    unittest.main()
