"""
MCP (Model Context Protocol) 客户端模块

提供通用的 MCP 客户端功能，用于连接和调用 MCP 服务器。
基于 fastmcp 实现，支持异步操作。

Example:
    >>> from gsuid_core.ai_core.mcp import MCPClient
    >>> client = MCPClient(
    ...     name="MiniMax",
    ...     command="uvx",
    ...     args=["minimax-coding-plan-mcp"],
    ...     env={"MINIMAX_API_KEY": "your_key"},
    ... )
    >>> tools = await client.list_tools()
    >>> result = await client.call_tool("web_search", {"query": "Python"})
"""

from gsuid_core.ai_core.mcp.client import MCPClient, MCPToolInfo, MCPToolResult

__all__ = ["MCPClient", "MCPToolInfo", "MCPToolResult"]
