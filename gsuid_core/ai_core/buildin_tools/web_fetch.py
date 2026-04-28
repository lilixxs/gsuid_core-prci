"""
网页抓取工具模块

提供网页内容抓取并转换为 Markdown 格式的功能，供 AI Agent 调用。
使用 aiohttp 进行异步 HTTP 请求，使用 markdownify 将 HTML 转换为 Markdown。
"""

from pydantic_ai import RunContext

from gsuid_core.ai_core.models import ToolContext
from gsuid_core.ai_core.register import ai_tools
from gsuid_core.ai_core.web_fetch import fetch_webpage_as_markdown


@ai_tools(category="buildin")
async def web_fetch_tool(
    ctx: RunContext[ToolContext],
    url: str,
) -> str:
    """
    网页抓取工具

    抓取指定 URL 的网页内容，将其转换为 Markdown 格式返回。
    适用于需要获取网页详细内容的场景，如阅读文章、获取文档等。

    Args:
        ctx: 工具执行上下文
        url: 要抓取的网页 URL，必须以 http:// 或 https:// 开头

    Returns:
        网页内容的 Markdown 格式文本

    Example:
        >>> content = await web_fetch_tool(ctx, "https://example.com")
        >>> print(content)
    """
    try:
        result = await fetch_webpage_as_markdown(url=url)
        return result
    except ValueError as e:
        return f"抓取失败: {e}"
