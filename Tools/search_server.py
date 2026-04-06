# ============================================================
# search_server.py
# STABLE MCP SERVER using Tavily API
# ============================================================

import importlib
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search")


def _try_load_dotenv() -> None:
    """Load .env values if python-dotenv is installed."""
    try:
        load_dotenv = getattr(importlib.import_module("dotenv"), "load_dotenv")
        load_dotenv()
    except Exception:
        return


def _get_tavily_client():
    """Create Tavily client from env var TAVILY_API_KEY."""
    _try_load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None, "TAVILY_API_KEY is missing. Add it in .env or environment variables."

    try:
        TavilyClient = getattr(importlib.import_module("tavily"), "TavilyClient")
        return TavilyClient(api_key=api_key), ""
    except Exception as exc:
        return None, f"Tavily client initialization error: {exc}"

@mcp.tool()
def search_web(query: str) -> str:
    """Search the web for real-time information.
    Use this for factual questions, historical data, or general lookups."""
    tavily, client_error = _get_tavily_client()
    if tavily is None:
        return f"Search error: {client_error}"

    try:
        # depth="basic" is faster and costs 1 credit
        response = tavily.search(query=query, search_depth="basic", max_results=3)
        results = response.get('results', [])
        
        if not results:
            return f"No results found for: '{query}'"
            
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n    {r['content']}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"Search error: {e}"

@mcp.tool()
def search_news(query: str) -> str:
    """Search for the latest news articles on a topic.
    Use this for recent events, announcements, or developments within the last month."""
    tavily, client_error = _get_tavily_client()
    if tavily is None:
        return f"News search error: {client_error}"

    try:
        # topic="news" triggers Tavily's news-specific crawler
        response = tavily.search(query=query, topic="news", search_depth="basic", max_results=3)
        results = response.get('results', [])
        
        if not results:
            return f"No news found for: '{query}'"
            
        return "\n\n".join([
            f"[{i+1}] {r['title']}\n"
            f"    Date: {r.get('published_date', 'Recent')}\n"
            f"    Content: {r['content']}\n"
            f"    Source: {r.get('url', 'Unknown')}"
            for i, r in enumerate(results)
        ])
    except Exception as e:
        return f"News search error: {e}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
