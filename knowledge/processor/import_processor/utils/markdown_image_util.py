"""图片节点使用的 Markdown 文本工具，不负责读取图片或调用外部服务。

处理顺序：normalize_html_images 统一正文中的图片语法 → image_references 扫描引用
→ 图片节点读取、上传和生成摘要 → transform_prose 替换正文。
共用正文/代码分离逻辑，避免教程代码里的示例图片路径被当作真实资源。
"""

import re
from html.parser import HTMLParser
from urllib.parse import quote


def transform_prose(content, transform, code_transform=None):
    """逐行区分代码与正文，再把片段交给相应回调，最后拼回完整文本。

    transform 接收正文片段并返回替换后的字符串；code_transform 处理代码片段，
    默认原样返回。回调也可以只收集信息、返回原文，因此扫描和替换能共用此函数。
    这是面向当前图片处理的轻量扫描器，并非完整 Markdown 语法解析器。
    """
    output, fence = [], None
    code_transform = code_transform or (lambda code: code)
    # 保留换行符，避免拼接后改变段落和代码排版；fence 记录当前未闭合的围栏。
    for line in content.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            # 已进入代码块：同类且长度足够的独立围栏行才能关闭它。
            output.append(code_transform(line))
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
            output.append(code_transform(line))
        elif line.startswith(("    ", "\t")):
            # 四个空格或 tab 开头按缩进代码处理，保留其中的示例语法。
            output.append(code_transform(line))
        else:
            # 普通行继续分离反引号包裹的行内代码，cursor 追踪尚未处理的位置。
            cursor = 0
            for match in re.finditer(r"(`+).*?\1", line):
                output.append(transform(line[cursor:match.start()]))
                output.append(code_transform(match[0]))
                cursor = match.end()
            output.append(transform(line[cursor:]))
    return "".join(output)


class _ImageTag(HTMLParser):
    """内部辅助解析器：只提取 img 属性，交给标准库处理引号与 HTML 实体。"""
    def __init__(self):
        """每次解析单个标签时创建新对象，避免不同图片属性互相残留。"""
        super().__init__(convert_charrefs=True)
        self.attributes = {}

    def handle_starttag(self, tag, attrs):
        """HTMLParser.feed 遇到起始标签时自动调用；attrs 为属性名/值对列表。"""
        if tag.lower() == "img":
            self.attributes = dict(attrs)


def normalize_html_images(content):
    """将正文中的 HTML img 转为 Markdown 图片，返回转换后的全文。"""
    def replace(match):
        """re.sub 的替换回调：无 src 时保留原标签，否则生成统一的图片引用。"""
        parser = _ImageTag()
        parser.feed(match[0])
        src = parser.attributes.get("src")
        if not src:
            return match[0]
        # 替换说明文字中的方括号，编码路径中的空格/括号，防止破坏 Markdown 边界。
        alt = (parser.attributes.get("alt") or "图片").replace("[", "（").replace("]", "）")
        return f"![{alt}](<{quote(src, safe='/:?=&%#')}>)"
    return transform_prose(content, lambda text: re.sub(r"<img\b[^>]*>", replace, text, flags=re.I))


def image_references(content, pattern):
    """只收集正文中匹配的图片引用；返回正则 Match 列表，不修改全文。

    Match 的位置相对于回调收到的正文片段，调用方应读取匹配内容/分组，
    不能把其偏移量直接当作整篇文档的位置。
    """
    references = []
    def collect(text):
        """收集当前正文片段的匹配项，并原样返回片段以保持内容不变。"""
        # 闭包引用外层列表以收集结果；返回原文满足 transform_prose 的回调契约。
        references.extend(pattern.finditer(text))
        return text
    transform_prose(content, collect)
    return references
