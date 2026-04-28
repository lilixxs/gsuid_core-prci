"""
MiniMax Web Search 模块

提供基于 MiniMax MCP 的 web 搜索功能。
通过 MCP 客户端连接 MiniMax MCP 服务器，调用其搜索工具。

MiniMax MCP 服务器通过 uvx 启动，需要配置 MINIMAX_API_KEY 等环境变量。
支持 api_key 池配置，实现自动轮询和失败重试。
"""

import random
from typing import Optional

from gsuid_core.logger import logger
from gsuid_core.data_store import get_res_path
from gsuid_core.ai_core.mcp import MCPClient
from gsuid_core.ai_core.resource import MISC_PATH
from gsuid_core.ai_core.configs.ai_config import minimax_config


def _get_api_key_pool() -> list[str]:
    """
    获取 api_key 池

    Returns:
        api_key 列表，如果是单个字符串则转换为单元素列表
    """
    api_key_data = minimax_config.get_config("api_key").data

    if isinstance(api_key_data, list):
        return [k for k in api_key_data if k]
    elif isinstance(api_key_data, str) and api_key_data:
        return [api_key_data]
    else:
        return []


def _select_api_key(api_key_pool: list[str]) -> Optional[str]:
    """
    从 api_key 池中选择一个 api_key（随机选择）

    Args:
        api_key_pool: api_key 列表

    Returns:
        选中的 api_key，如果池为空则返回 None
    """
    if not api_key_pool:
        return None
    return random.choice(api_key_pool)


def _build_mcp_client(api_key: str) -> MCPClient:
    """
    根据指定 api_key 构建 MiniMax MCP 客户端

    Args:
        api_key: MiniMax API Key

    Returns:
        配置好的 MCPClient 实例
    """
    env: dict[str, str] = {
        "MINIMAX_API_KEY": api_key,
        "MINIMAX_API_HOST": minimax_config.get_config("api_host").data or "https://api.minimaxi.com",
        "MINIMAX_API_RESOURCE_MODE": minimax_config.get_config("resource_mode").data or "url",
    }

    base_path = str(get_res_path(MISC_PATH / "minimax"))
    if base_path:
        env["MINIMAX_MCP_BASE_PATH"] = base_path

    return MCPClient(
        name="MiniMax",
        command="uvx",
        args=["minimax-coding-plan-mcp"],
        env=env,
    )


async def _do_minimax_search(
    query: str,
    max_results: int,
    api_key: str,
) -> list[dict]:
    """
    使用指定 api_key 执行 MiniMax MCP 搜索

    Args:
        query: 搜索查询关键词
        max_results: 最大返回结果数量
        api_key: MiniMax API Key

    Returns:
        搜索结果列表
    """
    client = _build_mcp_client(api_key)

    result = await client.call_tool(
        tool_name="web_search",
        arguments={
            "query": query,
            "max_results": max_results,
        },
    )

    if result.is_error:
        raise RuntimeError(f"MCP 搜索失败: {result.text}")

    # 调试日志：记录 MCP 返回的原始内容
    raw_text = result.text
    logger.debug(f"🌐 [WebSearch][MiniMax] MCP 返回原始内容 (长度={len(raw_text)}): {raw_text[:500]}...")

    return _parse_search_results(raw_text, max_results)


async def minimax_search(
    query: str,
    max_results: Optional[int] = None,
) -> list[dict]:
    """
    使用 MiniMax MCP 进行 web 搜索

    支持 api_key 池配置，会自动轮询尝试不同的 api_key。

    Args:
        query: 搜索查询关键词
        max_results: 最大返回结果数量，默认10条

    Returns:
        搜索结果列表，每条包含 title、url、content、score 等字段

    Example:
        >>> results = await minimax_search("Python 教程")
        >>> for r in results:
        ...     print(r["title"], r["url"])
    """
    api_key_pool = _get_api_key_pool()

    if not api_key_pool:
        logger.warning("🌐 [WebSearch][MiniMax] API Key 未配置，跳过搜索")
        return []

    if max_results is None:
        max_results = 10

    # 记录已尝试的 api_key，避免重复尝试
    tried_keys: set[str] = set()

    while len(tried_keys) < len(api_key_pool):
        api_key = _select_api_key([k for k in api_key_pool if k not in tried_keys])
        if not api_key:
            break

        tried_keys.add(api_key)

        try:
            results = await _do_minimax_search(
                query=query,
                max_results=max_results,
                api_key=api_key,
            )

            logger.info(f"🌐 [WebSearch][MiniMax] 搜索: {query}, 返回 {len(results)} 条结果")
            return results

        except Exception:
            logger.warning(f"🌐 [WebSearch][MiniMax] api_key ...{api_key[-4:]} 失败，尝试下一个")
            continue

    logger.error("🌐 [WebSearch][MiniMax] 所有 api_key 均失败")
    return []


def _extract_str(item: dict, *keys: str, default: str = "") -> str:
    """
    从字典中按优先级提取字符串值

    依次尝试 keys 中的键，返回第一个存在的非空值。
    避免使用 dict.get() 兜底语法。

    Args:
        item: 目标字典
        keys: 按优先级排列的键名
        default: 所有键都不存在时的默认值

    Returns:
        提取到的字符串值
    """
    for key in keys:
        if key in item and isinstance(item[key], str) and item[key]:
            return item[key]
    return default


def _extract_float(item: dict, key: str, default: float = 0.0) -> float:
    """
    从字典中提取浮点值

    Args:
        item: 目标字典
        key: 键名
        default: 键不存在时的默认值

    Returns:
        提取到的浮点值
    """
    if key in item and isinstance(item[key], (int, float)):
        return float(item[key])
    return default


def _extract_items_from_dict(data: dict) -> list:
    """
    从字典中提取结果列表

    尝试常见的键名 "results"、"data"、"organic"、"items"、"hits"。

    Args:
        data: 可能包含结果列表的字典

    Returns:
        结果列表，如果未找到则返回空列表
    """
    for key in ("results", "data", "organic", "items", "hits"):
        if key in data and isinstance(data[key], list):
            return data[key]
    return []


def _parse_search_result_item(item: dict) -> dict:
    """
    解析单条搜索结果

    Args:
        item: 单条搜索结果字典

    Returns:
        标准化的搜索结果字典
    """
    return {
        "title": _extract_str(item, "title"),
        "url": _extract_str(item, "url", "link"),
        "content": _extract_str(item, "content", "description", "snippet"),
        "score": _extract_float(item, "score"),
    }


def _parse_search_results(raw_text: str, max_results: int) -> list[dict]:
    """
    解析 MiniMax MCP 返回的搜索结果文本

    MiniMax MCP 可能返回以下格式：
    1. JSON 数组: [{"title": ..., "url": ..., ...}, ...]
    2. JSON 对象含 results/data 键: {"results": [...]}
    3. JSON 对象含单条结果: {"title": ..., "url": ..., ...}
    4. 纯文本/Markdown 格式的搜索结果

    Args:
        raw_text: MCP 返回的原始文本
        max_results: 最大返回结果数量

    Returns:
        解析后的搜索结果列表
    """
    import json

    results: list[dict] = []

    # 尝试解析为 JSON
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        data = None

    if isinstance(data, list):
        # JSON 数组格式
        for item in data[:max_results]:
            if isinstance(item, dict):
                results.append(_parse_search_result_item(item))
        if results:
            return results

    elif isinstance(data, dict):
        # JSON 对象格式：尝试提取结果列表
        items = _extract_items_from_dict(data)
        if items:
            for item in items[:max_results]:
                if isinstance(item, dict):
                    results.append(_parse_search_result_item(item))
            if results:
                return results

        # JSON 对象但没有 results/data 键，可能是单条结果
        if "title" in data or "url" in data or "content" in data:
            results.append(_parse_search_result_item(data))
            return results

        # JSON 对象有其他结构，尝试递归查找列表
        for value in data.values():
            if isinstance(value, list) and value:
                for item in value[:max_results]:
                    if isinstance(item, dict):
                        results.append(_parse_search_result_item(item))
                if results:
                    return results

    # 无法解析为结构化数据，将原始文本作为单条结果返回
    if raw_text.strip():
        results.append(
            {
                "title": "MiniMax 搜索结果",
                "url": "",
                "content": raw_text,
                "score": 0.0,
            }
        )

    return results


async def minimax_search_with_context(
    query: str,
    max_results: int = 5,
) -> dict:
    """
    使用 MiniMax MCP 进行带上下文的搜索

    Args:
        query: 搜索查询关键词
        max_results: 最大返回结果数量，默认5条

    Returns:
        包含 results(结果列表) 和 answer(AI摘要) 的字典
    """
    results = await minimax_search(query=query, max_results=max_results)
    return {"results": results, "answer": None}
