"""Markdown/HTML 表格进入 RAG 前的处理工具。

本工具统一保留四种演进方案的实现设计，但当前只启用方案三。这样节点层只依赖
``MarkdownTableUtil.process(markdown)``，将来切换方案或增加动态路由时无需改动
``DocumentSplitNode``。

方案一：表格隔离保护法（仅设计，未实现）
------------------------------------------
适用于不会超过切片上限的小型标准 Markdown 管道表格。实现时可扫描连续的
``| ... |`` 行，把整张表作为原子块从普通正文中隔离，正文交给通用切分器，表格直接
成为独立 chunk。优点是简单；局限是大表格仍会超限，而且不能处理 HTML 表格。
未来如果实现，建议提供 ``isolate_markdown_tables(markdown)``，返回带原文顺序标记的
正文块和表格块，不能分别返回两个无位置信息的列表，否则重新组装时会打乱顺序。

方案二：表头续传切分法（仅设计，未实现）
------------------------------------------
适用于数据行很多、语法规范的 Markdown 管道表格。实现时提取“表头 + 分隔线”，再按
字符预算或数据行数对表体分批，每个子表都重新带上表头。相比固定行数，更推荐按
``max_content_length - len(header)`` 动态装箱，否则某些单行很长时仍可能超限。
它解决了表头丢失和大表容量问题，但输出仍是二维管道语法，向量检索质量通常不如
自然语言。未来接口可命名为 ``split_markdown_table_with_header``。

方案三：降维转译法（当前实现）
--------------------------------
适用于标准 Markdown 管道表格和 MinerU 生成的常规 HTML 表格；HTML 支持
``rowspan`` 和 ``colspan``。处理分三阶段：

1. 矩阵投影：使用 BeautifulSoup 解析 HTML，把跨行单元格向下填充、跨列单元格向右
   填充，得到每行列数一致的二维数组；
2. 意图嗅探：区分标准表头表、左上角空置的交叉表和无表头的两列 K-V 表；
3. 语义重构：按类型生成自包含的自然语言，让单行被独立召回时仍带有列头、行头。

示例——标准表头表::

    功能 | 量程 | 精度
    直流电压 | 20V | ±0.5%

输出::

    - 【功能:直流电压，量程:20V，精度:±0.5%】

示例——交叉表::

    （空） | 良好 | 较弱
    9V电池 | >8.2V | 7.2至8.2V

输出::

    - 9V电池的良好标准为>8.2V
    - 9V电池的较弱标准为7.2至8.2V

示例——K-V 表::

    输入阻抗 | >1MΩ

输出::

    - 输入阻抗：>1MΩ

方案四：VLM 视觉语言模型降维（仅设计，未实现）
------------------------------------------------
适用于嵌套表格、多级不规则表头、跨页断裂等确定性算法难以可靠理解的结构。实现时先
用 Playwright 等无头浏览器渲染并截图，再调用 VLM 输出约束格式的自然语言键值描述。
生产实现还必须补充超时、重试、成本控制、图片清理、响应校验和脱敏日志，不能在本
工具中直接硬编码模型或密钥。未来可增加复杂度嗅探：发现嵌套 table、异常深 rowspan
或矩阵冲突时路由到方案四，其余 90% 常规表格仍走低成本的方案三。
"""

import logging
import re
from typing import Literal

from bs4 import BeautifulSoup, Tag


TableType = Literal["header_table", "cross_table", "kv_table", "unknown"]


class MarkdownTableUtil:
    """通过方案三把管道表格和 HTML 表格转成一维自然语言。"""

    name = "markdown_table_util"
    _table_pattern = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.I | re.S)
    _markdown_table_pattern = re.compile(
        r"(?m)(?:^[ \t]*\|[^\n]*\|[ \t]*(?:\n|$)){2,}"
    )
    _markdown_separator_cell_pattern = re.compile(r"^:?-{3,}:?$")
    _fence_pattern = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
    _logger = logging.getLogger(f"import.{name}")

    @classmethod
    def process(cls, markdown: str) -> str:
        """转译 Markdown 中的管道表格和完整 HTML 表格，其他内容保持原顺序。

        这是节点层唯一需要调用的入口。单张表解析失败时保留该表原文，并继续处理
        文档中的其他表格；日志只记录异常类型和表格序号，不打印可能敏感的表格内容。
        """
        # 代码围栏常用于展示 HTML/Markdown 表格源码，不能把示例误当作业务数据转译。
        result: list[str] = []
        plain_lines: list[str] = []
        fence_lines: list[str] = []
        active_fence: tuple[str, int] | None = None
        table_counter = [0]

        def flush_plain() -> None:
            if plain_lines:
                result.append(cls._process_table_text("".join(plain_lines), table_counter))
                plain_lines.clear()

        for line in markdown.splitlines(keepends=True):
            line_without_ending = line.rstrip("\r\n")
            if active_fence is not None:
                fence_lines.append(line)
                if cls._is_closing_fence(line_without_ending, active_fence):
                    result.append("".join(fence_lines))
                    fence_lines.clear()
                    active_fence = None
                continue

            opening_fence = cls._match_opening_fence(line_without_ending)
            if opening_fence is not None:
                flush_plain()
                active_fence = opening_fence
                fence_lines.append(line)
            else:
                plain_lines.append(line)

        # splitlines(keepends=True) 不会丢换行；未闭合围栏也按原文整体保留。
        if fence_lines:
            result.append("".join(fence_lines))
        flush_plain()
        return "".join(result)

    @classmethod
    def _process_table_text(cls, text: str, table_counter: list[int]) -> str:
        """只处理代码围栏外的一段文本，并维持 HTML/Markdown 表格原始位置。"""

        def replace_html_table(match: re.Match[str]) -> str:
            return cls._replace_table(
                match.group(0),
                table_kind="html",
                table_counter=table_counter,
            )

        def replace_markdown_table(match: re.Match[str]) -> str:
            raw_table = match.group(0)
            # 正则会吞掉表格末尾换行，单独保留可避免表格与后续正文粘连。
            table_without_ending = raw_table.rstrip("\r\n")
            line_ending = raw_table[len(table_without_ending) :]
            return cls._replace_table(
                table_without_ending,
                table_kind="markdown",
                table_counter=table_counter,
            ) + line_ending

        html_processed = cls._table_pattern.sub(replace_html_table, text)
        return cls._markdown_table_pattern.sub(replace_markdown_table, html_processed)

    @classmethod
    def _replace_table(
        cls,
        raw_table: str,
        *,
        table_kind: Literal["html", "markdown"],
        table_counter: list[int],
    ) -> str:
        """统一执行两种语法的方案三转译，并在失败时无损回退原表。"""
        table_index = table_counter[0]
        table_counter[0] += 1
        try:
            if table_kind == "html":
                linearized = cls.linearize_html_table(raw_table)
            else:
                grid, has_header_cells = cls.markdown_table_to_grid(raw_table)
                table_type = cls.detect_table_type(
                    grid,
                    has_header_cells=has_header_cells,
                )
                linearized = cls.linearize_grid(grid, table_type)
        except (TypeError, ValueError) as error:
            cls._logger.warning(
                "表格降维转译失败，已保留原文: table_index=%d, table_kind=%s, "
                "error_type=%s",
                table_index,
                table_kind,
                type(error).__name__,
            )
            return raw_table

        if not linearized:
            cls._logger.warning(
                "表格没有可转译数据，已保留原文: table_index=%d, table_kind=%s",
                table_index,
                table_kind,
            )
            return raw_table
        return "\n".join(["【表格转译】", linearized, "【表格转译结束】"])

    @classmethod
    def linearize_html_table(cls, html_table: str) -> str:
        """依次执行矩阵投影、意图嗅探和语义重构。

        方案三只处理单层常规表格。嵌套表格如果勉强按普通行解析会把内外层数据混在
        一起，因此当前明确报错并由 ``process`` 保留原文；未来应路由到方案四。
        """
        soup = BeautifulSoup(html_table, "html.parser")
        table = soup.find("table")
        if not isinstance(table, Tag):
            raise ValueError("未找到 table 元素")
        if table.find("table") is not None:
            raise ValueError("方案三不处理嵌套表格")

        grid = cls.html_table_to_grid(table)
        first_row = cls._direct_cells(cls._table_rows(table)[0]) if grid else []
        has_header_cells = any(cell.name.casefold() == "th" for cell in first_row)
        table_type = cls.detect_table_type(grid, has_header_cells=has_header_cells)
        return cls.linearize_grid(grid, table_type)

    @classmethod
    def html_table_to_grid(cls, table: Tag | str) -> list[list[str]]:
        """阶段一：展开 rowspan/colspan，返回列数一致的二维字符串矩阵。

        示例：第一行 ``<td rowspan="2">直流电压</td><td>20V</td>``，第二行只有
        ``<td>200V</td>``。投影后得到 ``[[直流电压, 20V], [直流电压, 200V]]``，
        第二行不再因为 HTML 省略跨行单元格而向左错位。

        Python 的 list 可以按需要扩容，所以这里无需预先猜测列数。每放一个单元格前
        先跳过已被上方 rowspan 占据的位置，再把值写入对应的矩形区域。
        """
        if isinstance(table, str):
            parsed_table = BeautifulSoup(table, "html.parser").find("table")
            if not isinstance(parsed_table, Tag):
                raise ValueError("未找到 table 元素")
            table = parsed_table

        rows = cls._table_rows(table)
        grid: list[list[str | None]] = [[] for _ in rows]
        for row_index, row in enumerate(rows):
            column_index = 0
            for cell in cls._direct_cells(row):
                # 上一行的 rowspan 可能已经占据当前位置，继续向右找第一个空格。
                while (
                    column_index < len(grid[row_index])
                    and grid[row_index][column_index] is not None
                ):
                    column_index += 1

                rowspan = cls._positive_span(cell.get("rowspan"))
                colspan = cls._positive_span(cell.get("colspan"))
                text = cell.get_text(" ", strip=True)

                # 跨行越过表格实际行数通常表示坏 HTML；静默截断会制造错误数据。
                if row_index + rowspan > len(grid):
                    raise ValueError("rowspan 超出表格行数")

                for row_offset in range(rowspan):
                    target_row = grid[row_index + row_offset]
                    required_width = column_index + colspan
                    if len(target_row) < required_width:
                        target_row.extend([None] * (required_width - len(target_row)))
                    for column_offset in range(colspan):
                        target_column = column_index + column_offset
                        if target_row[target_column] is not None:
                            raise ValueError("rowspan/colspan 投影位置发生冲突")
                        # colspan 使用相同文本向右填充，保证每个物理列都有上下文。
                        target_row[target_column] = text
                column_index += colspan

        width = max((len(row) for row in grid), default=0)
        # None 只表示 HTML 中没有对应单元格，统一转为空字符串方便后续序列化。
        return [
            [(cell or "") for cell in row + [None] * (width - len(row))]
            for row in grid
        ]

    @classmethod
    def markdown_table_to_grid(
        cls,
        markdown_table: str,
    ) -> tuple[list[list[str]], bool]:
        """把标准管道表格转成矩阵，后续复用方案三的嗅探和语义重构。

        标准表格的第二行 ``|---|---|`` 只描述对齐方式，不是业务数据，因此从矩阵
        删除并返回 ``has_header_cells=True``。没有分隔行的两列表则按 K-V 表处理。
        转义管道符 ``\\|`` 会还原为单元格正文中的普通 ``|``。
        """
        rows = [
            cls._split_markdown_row(line)
            for line in markdown_table.strip().splitlines()
            if line.strip()
        ]
        if not rows:
            return [], False

        width = max(len(row) for row in rows)
        grid = [row + [""] * (width - len(row)) for row in rows]
        has_separator = len(grid) > 1 and all(
            cls._markdown_separator_cell_pattern.fullmatch(cell.strip())
            for cell in grid[1]
        )
        if has_separator:
            del grid[1]
        return grid, has_separator

    @staticmethod
    def detect_table_type(
        grid: list[list[str]],
        *,
        has_header_cells: bool = False,
    ) -> TableType:
        """阶段二：根据矩阵形态嗅探标准表、交叉表或 K-V 表。

        判断顺序很重要：左上角为空且超过两列时优先认定为交叉表；两列表如果首行使用
        ``<th>``，仍属于标准表头表，否则按无表头 K-V 表处理；其他多列表默认使用
        第一行作为表头。这个规则是确定性的，遇到无法可靠推断的多级表头应交给未来的
        方案四，而不是在这里调用 LLM 猜测。
        """
        if not grid or not grid[0]:
            return "unknown"
        width = len(grid[0])
        first_cell = grid[0][0].strip()
        if not first_cell and width > 2:
            return "cross_table"
        if width == 2 and not has_header_cells:
            return "kv_table"
        return "header_table"

    @staticmethod
    def linearize_grid(grid: list[list[str]], table_type: TableType) -> str:
        """阶段三：按表格类型生成适合 embedding 的自包含自然语言。

        标准表的每个结果行重复全部列名；交叉表的每个结果句同时重复行头和列头；
        K-V 表直接生成“键：值”。这种有意的文本冗余能换取 chunk 被再次切开后仍具有
        完整语义，也是方案三相对简单“第 N 行：值列表”的核心提升。
        """
        if table_type == "unknown" or not grid:
            return ""

        results: list[str] = []
        if table_type == "header_table":
            headers = [header.strip() or f"第{index}列" for index, header in enumerate(grid[0], 1)]
            for row in grid[1:]:
                parts = [
                    f"{header}:{value.strip()}"
                    for header, value in zip(headers, row)
                    if value.strip()
                ]
                if parts:
                    results.append(f"- 【{'，'.join(parts)}】")

            # 只有表头、没有数据行时仍保留信息，避免转译后得到空字符串。
            if not results and any(header.strip() for header in grid[0]):
                results.append(f"- 【{'，'.join(headers)}】")

        elif table_type == "cross_table":
            column_headers = grid[0][1:]
            for row_index, row in enumerate(grid[1:], start=1):
                row_header = row[0].strip() or f"第{row_index}行"
                for column_header, value in zip(column_headers, row[1:]):
                    if value.strip():
                        header = column_header.strip() or "未命名列"
                        results.append(f"- {row_header}的{header}标准为{value.strip()}")

        elif table_type == "kv_table":
            for row in grid:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    results.append(f"- {row[0].strip()}：{row[1].strip()}")

        return "\n".join(results)

    @staticmethod
    def _positive_span(raw_value: object) -> int:
        """把缺失 span 视为 1；零、负数和非数字属于无法可靠投影的坏表格。"""
        if raw_value is None:
            return 1
        try:
            span = int(str(raw_value))
        except ValueError as error:
            raise ValueError("rowspan/colspan 不是整数") from error
        if span <= 0:
            raise ValueError("rowspan/colspan 必须是正整数")
        return span

    @staticmethod
    def _table_rows(table: Tag) -> list[Tag]:
        """返回当前 table 的行，排除未来可能出现的嵌套 table 行。"""
        return [
            row
            for row in table.find_all("tr")
            if row.find_parent("table") is table
        ]

    @staticmethod
    def _direct_cells(row: Tag) -> list[Tag]:
        """只读取当前 tr 的直接单元格，避免误收集单元格内部的嵌套结构。"""
        return [
            cell
            for cell in row.find_all(["th", "td"], recursive=False)
            if isinstance(cell, Tag)
        ]

    @staticmethod
    def _split_markdown_row(line: str) -> list[str]:
        """按未转义的管道符拆行，兼容单元格正文里的 ``\\|``。"""
        stripped = line.strip()
        if not stripped.startswith("|") or not stripped.endswith("|"):
            raise ValueError("Markdown 表格行必须以管道符开始和结束")
        inner = stripped[1:-1]
        return [
            cell.replace(r"\|", "|").strip()
            for cell in re.split(r"(?<!\\)\|", inner)
        ]

    @classmethod
    def _match_opening_fence(cls, line: str) -> tuple[str, int] | None:
        """记录代码围栏字符和长度，供表格扫描跳过示例代码。"""
        match = cls._fence_pattern.match(line)
        if match is None:
            return None
        marker = match.group(1)
        if marker[0] == "`" and "`" in match.group(2):
            return None
        return marker[0], len(marker)

    @classmethod
    def _is_closing_fence(cls, line: str, active_fence: tuple[str, int]) -> bool:
        """关闭围栏必须字符相同、长度足够且行尾没有其他内容。"""
        match = cls._fence_pattern.match(line)
        if match is None or match.group(2).strip():
            return False
        marker = match.group(1)
        character, length = active_fence
        return marker[0] == character and len(marker) >= length
