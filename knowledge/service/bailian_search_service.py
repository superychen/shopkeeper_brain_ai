"""【07】百炼 MCP 适配：发现工具契约后确定性调用，不交由模型自由选择工具。

网络节点 run → session 校验配置/建立传输/初始化/发现工具 → search 校验参数
→ call_tool → parse_pages → 标准网页列表 → 退出session清理连接。
示例中的pages/title/url/snippet是当前适配器支持的契约，不代表已验证目标服务。
工具inputSchema会在运行时读取；实际返回格式若不同，应报契约错误再适配，
不能吞掉解析异常后伪装成“网上没有搜索结果”。整个模块不抓取网页完整正文。
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import asyncio
import json
import re
from urllib.parse import urlsplit, urlunsplit

from knowledge.processor.query_processor.exceptions import QueryError
from knowledge.service.retrieval_runtime import MCP_SESSIONS


def parse_pages(result, item, limit):
    """【07.4】将MCP工具结果转换为网页证据，返回 (候选列表, 告警列表)。

    输入结构示例：result.structuredContent={"pages":[{"title":"手册",
    "url":"https://example.com/manual#top","snippet":"摘要"}]}，item="RS-12"。
    输出示例：([{title:"手册",url:"https://example.com/manual",snippet:"摘要",
    item_names:["RS-12"],rank:1,source_type:"web",published_at:None,...}],[])。
    没有structuredContent时逐个解析text块，允许多个独立JSON对象，不盲目拼接文本。
    pages=[] 是合法空结果；缺少pages、全部网页损坏等是 mcp_response_invalid。
    部分条目损坏可返回有效条目并附 invalid_web_items，工具isError则直接失败。
    """
    if getattr(result, "isError", False):
        raise QueryError("mcp_tool_error", "搜索工具返回错误")
    structured = getattr(result, "structuredContent", None)
    # 有结构化内容就优先采用；结构化字段损坏时不能偷偷改读另一份文本掩盖契约变化。
    try:
        payloads = [structured] if structured is not None else [
            json.loads(block.text) for block in result.content if getattr(block, "type", None) == "text"]
        if not payloads or any(not isinstance(p, dict) or not isinstance(p.get("pages"), list) for p in payloads):
            raise ValueError("不支持的网页响应契约")
        pages = [page for payload in payloads for page in payload["pages"]]
        # 展平多个JSON块里的pages；seen用于当前商品内的URL去重，invalid只统计坏条目。
        hits, seen, invalid = [], set(), 0
        for page in pages:
            try:
                if not isinstance(page, dict) or any(not isinstance(page.get(k), str) for k in ("title", "url", "snippet")):
                    raise ValueError()
                url = urlsplit(page["url"])
                # 只接受可引用的HTTP(S)网址，拒绝带用户名/密码的链接；这里只解析，不发请求。
                if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
                    raise ValueError()
                normalized = urlunsplit((url.scheme, url.netloc.lower(), url.path, url.query, ""))
                # 去fragment和统一主机大小写，但保留路径和query：?id=1与?id=2可能是不同正文。
                if normalized in seen:
                    continue
                seen.add(normalized)
                hits.append(dict(source_type="web", item_names=[item], title=page["title"][:1024],
                    url=normalized, snippet=page["snippet"][:8000], rank=len(hits)+1,
                    provider="aliyun_bailian", retrieved_at=datetime.now(timezone.utc).isoformat(),
                    published_at=page.get("published_at") if isinstance(page.get("published_at"), str) else None))
            except (ValueError, TypeError):
                invalid += 1
        # 长度截取限制进入state的标题/摘要；retrieved_at是本次接收时间，不是发布时间。
        # 服务没有提供字符串published_at则保留None，不能用当前时间假装资料刚发布。
        if pages and not hits:
            raise ValueError("所有网页均无效")
        return hits[:limit], (["invalid_web_items"] if invalid else [])
    except (ValueError, TypeError, AttributeError) as exc:
        raise QueryError("mcp_response_invalid", "网络响应格式不匹配", exc) from exc


class BailianSearchService:
    """将提供方协议细节封装起来；节点只关心会话句柄和网页候选。

    配置决定唯一目标端点/工具，用户问题只能成为query参数，不能指定任意工具或headers。
    不缓存问题与网页正文，每请求一个会话；同请求的多个商品共用一次工具发现结果。
    """
    def __init__(self, config):
        """保存RetrievalConfig；此处不解包密钥，也不在构造阶段联网。"""
        self.config = config

    @asynccontextmanager
    async def session(self):
        """【07.1】建立一个已初始化的搜索会话，yield (ClientSession,inputSchema)。

        调用示例：async with service.session() as connection:
                     await service.search(connection,"RS-12","怎么用？")
        先校验HTTPS百炼地址、独立凭据和工具名，再按HTTP→transport→ClientSession
        顺序进入上下文；退出按相反顺序清理。MCP 2.1.1传输返回(read,write)二元组。
        initialize完成协议握手，list_tools取得当前可用工具，不通过名字猜参数。
        缺配置抛mcp_configuration_missing；目标工具不在列表抛mcp_tool_schema_mismatch。
        yield期间调用方使用会话处理多个商品；即使其失败/取消，上下文仍负责收尾。
        """
        cfg = self.config
        url = urlsplit(cfg.mcp_url)
        # 地址来自部署配置，不接受控制台页面、非百炼主机、额外认证URL或查询参数。
        # 保持空配置可构建图，在真正执行本路时单独失败，避免影响本地两路。
        if (not cfg.api_key.get_secret_value() or not cfg.tool_name or url.scheme != "https"
                or url.hostname != "dashscope.aliyuncs.com" or url.username or url.password
                or url.query or url.fragment or not url.path.startswith("/api/v1/mcps/")):
            raise QueryError("mcp_configuration_missing", "请配置目标百炼服务外部地址、工具名和凭据")
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
        import httpx2
        # 只有这里解包SecretStr用于Bearer头，禁止把headers/config/客户端完整对象打到日志。
        # follow_redirects=False防止认证头随着重定向发往其他主机；connect预算5秒。
        # MCP_SESSIONS限制跨请求会话数，外层节点web_timeout限制整个会话工作时长。
        async with MCP_SESSIONS.slot(), httpx2.AsyncClient(headers={"Authorization": "Bearer " + cfg.api_key.get_secret_value()},
                timeout=httpx2.Timeout(cfg.call_timeout, connect=5), follow_redirects=False) as http:
            async with streamable_http_client(cfg.mcp_url, http_client=http) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=cfg.call_timeout) as session:
                    await session.initialize()
                    catalog = await session.list_tools()
                    # next(...,None)取得指定名字的第一个工具；找不到时明确失败，不调用其他工具。
                    tool = next((t for t in catalog.tools if t.name == cfg.tool_name), None)
                    if tool is None:
                        raise QueryError("mcp_tool_schema_mismatch", "目标搜索工具不存在")
                    yield session, tool.inputSchema

    async def search(self, connection, item, query):
        """【07.2～07.3】按真实inputSchema校验参数，再执行一次搜索。

        connection是session()产出的二元组；item为标准商品名，query为当前完整问题。
        示例：item="RS-12"、query="怎么用？" → args.query="RS-12\n怎么用？"；
        schema声明count时再传count=3，否则只发送query。
        必填字段新增或count范围不兼容时，在外呼前抛mcp_tool_schema_mismatch。
        返回parse_pages的(网页列表,告警列表)，不发送history或HyDE假设，也不让模型
        自选工具。单次工具调用超时后不自动重试，避免请求实际完成却再次产生搜索费用。
        """
        # 最小脱敏检测阻止常见密钥/密码字段外发；不是完整敏感信息识别系统。
        if re.search(r"sk-[A-Za-z0-9]{16,}|(?:password|api[_-]?key|密码|密钥)\s*[:=：]", query, re.I):
            raise QueryError("web_query_requires_redaction", "问题含需脱敏内容")
        session, schema = connection
        args = {"query": f"{item}\n{query}"}
        if "count" in schema.get("properties", {}):
            args["count"] = self.config.web_results_per_item
        # 使用真实 inputSchema 校验，新增必填参数时明确失败，而不是猜测其含义。
        import jsonschema
        try:
            jsonschema.validate(args, schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError) as exc:
            raise QueryError("mcp_tool_schema_mismatch", "工具参数契约不兼容", exc) from exc
        async with asyncio.timeout(self.config.call_timeout):
            # 这是本模块真正发起搜索的位置；握手/list_tools与搜索调用是不同阶段。
            result = await session.call_tool(self.config.tool_name, args)
        return parse_pages(result, item, self.config.web_results_per_item)
