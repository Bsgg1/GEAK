import json
from pathlib import Path
from typing import Any

from minisweagent.tools.bash_command import BashCommand
from minisweagent.tools.str_replace_editor import str_replace_editor
from minisweagent.tools.submit import SubmitTool

current_dir = Path(__file__).resolve().parent
json_path = current_dir / "tools.json"
with open(json_path, encoding="utf-8") as f:
    _all_tools = json.load(f)


def get_tools_list() -> list:
    return list(_all_tools)


tools_list = _all_tools


class ToolRuntime:
    def __init__(self, rag_config: dict | None = None):
        self._mcp_bridges: list = []

        self._tool_table = {
            "bash": BashCommand(),
            "str_replace_editor": str_replace_editor(),
            "submit": SubmitTool(),
        }

        if rag_config:
            self._register_rag_mcp_tools(rag_config)

    def _register_rag_mcp_tools(self, rag_config: dict) -> None:
        try:
            from minisweagent.tools.mcp_bridge import MCPToolBridge
        except ImportError:
            return

        rag_bridge = MCPToolBridge("rag-mcp", timeout=120)
        self._mcp_bridges.append(rag_bridge)

        enable_subagent = rag_config.get("enable_subagent", False)
        if enable_subagent:
            from minisweagent.agents.subagent import RAGFilterSubAgent, SubAgentConfig

            subagent = RAGFilterSubAgent(SubAgentConfig(enabled=True))

            def _wrap_with_subagent(tool_callable, tool_name: str):
                def wrapper(**kwargs):
                    result = tool_callable(**kwargs)
                    output = result.get("output", "")
                    if output and result.get("returncode") == 0:
                        query = kwargs.get("topic") or kwargs.get("code_type") or ""
                        result["output"] = subagent.process(output, query=query)
                    return result
                return wrapper

            self._tool_table["rag_query"] = _wrap_with_subagent(rag_bridge.tool("query"), "query")
            self._tool_table["rag_optimize"] = _wrap_with_subagent(rag_bridge.tool("optimize"), "optimize")
        else:
            self._tool_table["rag_query"] = rag_bridge.tool("query")
            self._tool_table["rag_optimize"] = rag_bridge.tool("optimize")

    def set_env(self, env: dict[str, str]) -> None:
        env = dict(env)
        bash = self._tool_table.get("bash")
        if bash is not None:
            bash._env_override = env
        for bridge in self._mcp_bridges:
            bridge.set_env(env)

    def set_cwd(self, cwd: str | None) -> None:
        bash = self._tool_table.get("bash")
        if bash is not None:
            bash._cwd = cwd

    def get_tools_schema(self) -> list[dict]:
        """Return tool schemas only for tools registered in _tool_table."""
        return [t for t in _all_tools if t["name"] in self._tool_table]

    def dispatch(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        name = tool_call["name"]
        args = tool_call.get("arguments", {})

        if name not in self._tool_table:
            raise ValueError(f"Unknown tool: {name}")

        if name == "bash" and "command" not in args:
            args = {**args, "command": ""}

        return self._tool_table[name](**args)
