"""DocumentSplitNode Step 1/2 单元测试。"""

import unittest
from pathlib import Path

from knowledge.processor.import_processor.config import ImportConfig
from knowledge.processor.import_processor.exceptions import (
    ConfigurationError,
    StateFieldError,
)
from knowledge.processor.import_processor.nodes.document_split_node import (
    DocumentSplitNode,
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

    def test_process_does_not_publish_intermediate_sections_as_chunks(self) -> None:
        state = self._state("# 标题\n正文", chunks=["existing-result"])

        result = self.node.process(state)

        self.assertIs(result, state)
        self.assertEqual(result["chunks"], ["existing-result"])

    def test_base_node_preserves_classified_validation_error(self) -> None:
        with self.assertRaises(StateFieldError):
            self.node({"md_path": str(self.workspace / "manual.md")})


if __name__ == "__main__":
    unittest.main()
