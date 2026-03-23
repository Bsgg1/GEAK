"""
MCP server configuration registry.
"""

from pathlib import Path
from typing import Any

# Get the msa root directory (mcp-client is in mcp_tools/mcp-client)
MSA_ROOT = Path(__file__).parent.parent.parent.parent.parent

# MCP server configurations
MCP_SERVERS = {
    "rag-mcp": {
        "command": ["python3", "-m", "rag_mcp.server"],
        "cwd": str(MSA_ROOT / "mcp_tools" / "rag-mcp"),
        "env": {"PYTHONPATH": str(MSA_ROOT / "mcp_tools" / "rag-mcp" / "src")},
        "tools": ["query", "optimize"],
        "description": "GPU/ROCm/HIP knowledge base retrieval via hybrid search",
    },
}


def get_server_config(server_name: str) -> dict[str, Any]:
    """
    Get configuration for an MCP server.

    Args:
        server_name: Name of the MCP server

    Returns:
        Server configuration dict

    Raises:
        KeyError: If server not found
    """
    if server_name not in MCP_SERVERS:
        available = ", ".join(MCP_SERVERS.keys())
        raise KeyError(f"Unknown MCP server: {server_name}. Available servers: {available}")

    return MCP_SERVERS[server_name]


def list_servers() -> dict[str, str]:
    """
    List all available MCP servers with descriptions.

    Returns:
        Dict mapping server names to descriptions
    """
    return {name: config.get("description", "No description") for name, config in MCP_SERVERS.items()}
