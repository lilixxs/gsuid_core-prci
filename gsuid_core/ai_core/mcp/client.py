"""
MCP 客户端核心模块

提供通用的 MCP 客户端功能，用于连接和调用 MCP 服务器。
基于 fastmcp 实现，支持通过 stdio 方式连接 MCP 服务器。

设计原则：
- 每次调用时建立连接、执行操作、断开连接（无状态模式）
- 支持通过代码配置连接参数（command, args, env）
- 完全异步，兼容项目的 async 架构
"""

from typing import Any
from dataclasses import field, dataclass

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from mcp.types import TextContent, ImageContent, ResourceLink, EmbeddedResource
from gsuid_core.logger import logger


@dataclass
class MCPToolInfo:
    """MCP 工具信息"""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class MCPToolResult:
    """MCP 工具调用结果"""

    content: list[dict[str, Any]]
    is_error: bool = False

    @property
    def text(self) -> str:
        """提取所有文本内容并拼接"""
        texts: list[str] = []
        for item in self.content:
            if "type" in item and item["type"] == "text" and "text" in item:
                text_value = item["text"]
                if isinstance(text_value, str):
                    texts.append(text_value)
        return "\n".join(texts)


@dataclass
class MCPClient:
    """
    MCP 客户端

    通过 stdio 方式连接 MCP 服务器，提供工具列表查询和工具调用功能。
    每次操作独立建立连接，操作完成后自动断开。

    Args:
        name: MCP 服务器名称，用于日志标识
        command: 启动命令，如 "uvx", "npx", "python" 等
        args: 命令参数列表
        env: 环境变量字典

    Example:
        >>> client = MCPClient(
        ...     name="MiniMax",
        ...     command="uvx",
        ...     args=["minimax-coding-plan-mcp"],
        ...     env={"MINIMAX_API_KEY": "your_key"},
        ... )
        >>> tools = await client.list_tools()
        >>> result = await client.call_tool("web_search", {"query": "Python"})
    """

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)

    def _create_transport(self) -> StdioTransport:
        """创建 stdio 传输层"""
        return StdioTransport(
            command=self.command,
            args=self.args,
            env=self.env if self.env else None,
        )

    async def list_tools(self) -> list[MCPToolInfo]:
        """
        列出 MCP 服务器提供的所有工具

        Returns:
            工具信息列表

        Raises:
            连接或通信失败时抛出异常
        """
        transport = self._create_transport()
        client = Client(transport)

        logger.info(f"🔌 [MCP][{self.name}] 正在连接服务器并获取工具列表...")

        async with client:
            raw_tools = await client.list_tools()

        tools: list[MCPToolInfo] = []
        for tool in raw_tools:
            schema = tool.inputSchema
            tools.append(
                MCPToolInfo(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    input_schema=schema if schema else {},
                )
            )

        logger.info(f"🔌 [MCP][{self.name}] 获取到 {len(tools)} 个工具")
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """
        调用 MCP 服务器上的指定工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数字典

        Returns:
            工具调用结果

        Raises:
            连接或调用失败时抛出异常
        """
        transport = self._create_transport()
        client = Client(transport)

        logger.info(f"🔌 [MCP][{self.name}] 调用工具: {tool_name}, 参数: {arguments}")

        async with client:
            result = await client.call_tool(
                name=tool_name,
                arguments=arguments or {},
            )

        # 将 CallToolResult 转换为统一格式
        content_list: list[dict[str, Any]] = []
        for item in result.content:
            if isinstance(item, TextContent):
                content_list.append({"type": "text", "text": item.text})
            elif isinstance(item, ImageContent):
                content_list.append(
                    {
                        "type": "image",
                        "data": item.data,
                        "mimeType": item.mimeType,
                    }
                )
            elif isinstance(item, (ResourceLink, EmbeddedResource)):
                content_list.append({"type": "resource", "text": str(item)})
            else:
                content_list.append({"type": "text", "text": str(item)})

        tool_result = MCPToolResult(
            content=content_list,
            is_error=result.is_error,
        )

        logger.info(
            f"🔌 [MCP][{self.name}] 工具 {tool_name} 调用完成, "
            f"is_error={tool_result.is_error}, "
            f"内容长度={len(tool_result.text)}"
        )

        return tool_result
