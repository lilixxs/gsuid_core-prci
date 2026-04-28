"""
Web Search 公共 API 模块

提供统一的 web 搜索接口，根据用户配置自动选择搜索引擎（Tavily / Exa / MiniMax）。
外部模块应通过本模块的函数调用搜索，无需关心底层搜索引擎的实现细节。
"""

from gsuid_core.ai_core.configs.ai_config import ai_config

from .exa_search import exa_search
from .tavily_search import (
    tavily_search,
    tavily_search_with_context,
)
from .minimax_search import (
    minimax_search,
    minimax_search_with_context,
)


def _get_provider() -> str:
    """
    获取当前配置的搜索引擎提供方

    Returns:
        搜索引擎名称，如 "Tavily"、"Exa" 或 "MiniMax"
    """
    return ai_config.get_config("websearch_provider").data


async def web_search(
    query: str,
    max_results: int | None = None,
) -> list[dict]:
    """
    统一的 web 搜索接口

    根据用户配置的 websearch_provider 自动选择搜索引擎。
    支持 api_key 池配置，会自动轮询尝试不同的 api_key。

    Args:
        query: 搜索查询关键词
        max_results: 最大返回结果数量，默认由各搜索引擎配置决定

    Returns:
        搜索结果列表，每条包含 title、url、content、score 等字段

    Example:
        >>> results = await web_search("Python 教程")
        >>> for r in results:
        ...     print(r["title"], r["url"])
    """
    provider = _get_provider()

    if provider == "Exa":
        return await exa_search(query=query, max_results=max_results)

    if provider == "MiniMax":
        return await minimax_search(query=query, max_results=max_results)

    # 默认使用 Tavily
    return await tavily_search(query=query, max_results=max_results)


async def web_search_with_context(
    query: str,
    max_results: int = 5,
) -> dict:
    """
    统一的带上下文 web 搜索接口

    根据用户配置的 websearch_provider 自动选择搜索引擎。
    该方法会同时返回搜索结果和 AI 生成的摘要答案（如果搜索引擎支持）。

    Args:
        query: 搜索查询关键词
        max_results: 最大返回结果数量，默认5条

    Returns:
        包含 results(结果列表) 和 answer(AI摘要) 的字典
    """
    provider = _get_provider()

    if provider == "Exa":
        results = await exa_search(query=query, max_results=max_results)
        return {"results": results, "answer": None}

    if provider == "MiniMax":
        return await minimax_search_with_context(query=query, max_results=max_results)

    # 默认使用 Tavily
    return await tavily_search_with_context(query=query, max_results=max_results)
