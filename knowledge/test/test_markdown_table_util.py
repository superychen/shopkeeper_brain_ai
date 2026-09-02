"""MarkdownTableUtil 方案三的矩阵、嗅探和语义转译测试。"""

import unittest

from knowledge.processor.import_processor.utils.markdown_table_util import (
    MarkdownTableUtil,
)


class MarkdownTableUtilTest(unittest.TestCase):
    """覆盖三种表格意图以及跨行、跨列和失败降级场景。"""

    def test_header_table_repeats_headers_for_every_data_row(self) -> None:
        html = """<table>
<tr><th>功能</th><th>量程</th><th>精度</th></tr>
<tr><td>直流电压</td><td>20V</td><td>±0.5%</td></tr>
<tr><td>交流电压</td><td>600V</td><td>±1.2%</td></tr>
</table>"""

        result = MarkdownTableUtil.process(html)

        self.assertIn("- 【功能:直流电压，量程:20V，精度:±0.5%】", result)
        self.assertIn("- 【功能:交流电压，量程:600V，精度:±1.2%】", result)

    def test_cross_table_combines_row_and_column_headers(self) -> None:
        html = """<table>
<tr><td></td><td>良好</td><td>较弱</td><td>坏的</td></tr>
<tr><td>9V电池</td><td>&gt;8.2V</td><td>7.2至8.2V</td><td>&lt;7.2V</td></tr>
</table>"""

        result = MarkdownTableUtil.process(html)

        self.assertIn("- 9V电池的良好标准为>8.2V", result)
        self.assertIn("- 9V电池的较弱标准为7.2至8.2V", result)
        self.assertIn("- 9V电池的坏的标准为<7.2V", result)

    def test_two_column_table_without_th_is_treated_as_key_value(self) -> None:
        html = """<table>
<tr><td>输入阻抗</td><td>&gt;1MΩ</td></tr>
<tr><td>显示</td><td>3½位液晶显示</td></tr>
</table>"""

        result = MarkdownTableUtil.process(html)

        self.assertIn("- 输入阻抗：>1MΩ", result)
        self.assertIn("- 显示：3½位液晶显示", result)

    def test_matrix_projection_physically_fills_rowspan_and_colspan(self) -> None:
        html = """<table>
<tr><td rowspan="2">直流电压</td><td colspan="2">20V</td></tr>
<tr><td>0.01V</td><td>±0.5%</td></tr>
</table>"""

        grid = MarkdownTableUtil.html_table_to_grid(html)

        self.assertEqual(
            grid,
            [
                ["直流电压", "20V", "20V"],
                ["直流电压", "0.01V", "±0.5%"],
            ],
        )

    def test_two_column_table_with_th_is_still_a_header_table(self) -> None:
        html = """<table>
<tr><th>型号</th><th>量程</th></tr>
<tr><td>A1</td><td>20V</td></tr>
</table>"""

        result = MarkdownTableUtil.process(html)

        self.assertIn("- 【型号:A1，量程:20V】", result)
        self.assertNotIn("- 型号：量程", result)

    def test_standard_markdown_table_reuses_semantic_reconstruction(self) -> None:
        markdown = """| 功能 | 量程 | 精度 |
|---|---|---|
| 直流电压 | 20V | ±0.5% |
| 交流电压 | 600V | ±1.2% |"""

        result = MarkdownTableUtil.process(markdown)

        self.assertIn("- 【功能:直流电压，量程:20V，精度:±0.5%】", result)
        self.assertIn("- 【功能:交流电压，量程:600V，精度:±1.2%】", result)
        self.assertNotIn("|---|", result)

    def test_markdown_key_value_table_without_separator_is_supported(self) -> None:
        markdown = "| 输入阻抗 | >1MΩ |\n| 显示 | 3½位液晶显示 |"

        result = MarkdownTableUtil.process(markdown)

        self.assertIn("- 输入阻抗：>1MΩ", result)
        self.assertIn("- 显示：3½位液晶显示", result)

    def test_tables_inside_code_fence_are_not_translated(self) -> None:
        markdown = """```markdown
| A | B |
|---|---|
| 1 | 2 |
```

```html
<table><tr><td>键</td><td>值</td></tr></table>
```"""

        result = MarkdownTableUtil.process(markdown)

        self.assertEqual(result, markdown)
        self.assertNotIn("【表格转译】", result)

    def test_multiple_tables_keep_surrounding_markdown_order(self) -> None:
        markdown = (
            "前文\n<table><tr><td>键1</td><td>值1</td></tr></table>\n"
            "中间\n<table><tr><td>键2</td><td>值2</td></tr></table>\n后文"
        )

        result = MarkdownTableUtil.process(markdown)

        self.assertLess(result.index("前文"), result.index("- 键1：值1"))
        self.assertLess(result.index("- 键1：值1"), result.index("中间"))
        self.assertLess(result.index("中间"), result.index("- 键2：值2"))
        self.assertLess(result.index("- 键2：值2"), result.index("后文"))

    def test_invalid_span_keeps_original_table_and_logs_warning(self) -> None:
        html = '<table><tr><td rowspan="bad">数据</td></tr></table>'

        with self.assertLogs(MarkdownTableUtil._logger, level="WARNING"):
            result = MarkdownTableUtil.process(html)

        self.assertEqual(result, html)

    def test_plain_markdown_without_table_is_unchanged(self) -> None:
        markdown = "# 标题\n\n普通正文\n\n- 列表一\n- 列表二"

        result = MarkdownTableUtil.process(markdown)

        self.assertEqual(result, markdown)


if __name__ == "__main__":
    unittest.main()
