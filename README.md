# Mini SWE Agent

A minimal AI coding agent powered by LLM tool calling and Bash commands. Features optional RAG knowledge retrieval via MCP (Model Context Protocol) for GPU/ROCm/HIP optimization tasks.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Interactive REPL (default: confirm mode)
mini

# Run with a specific task
mini -t "fix the bug in main.py"

# Auto-execute mode (no confirmation needed)
mini -y

# Use Textual TUI
mini -v

# Enable RAG knowledge retrieval
mini -c mini_rag

# Custom config
mini -c /path/to/config.yaml
```

## RAG MCP Integration

Optional GPU/ROCm/HIP knowledge base retrieval via a dedicated MCP server. Uses hybrid search (embedding + BM25 + RRF fusion + BGE reranker).

### Enable RAG

```bash
# Use the built-in mini_rag config
mini -c mini_rag -t "optimize this HIP kernel for MI300X"
```

Or add `rag:` to any config YAML:

```yaml
# RAG MCP toggle: comment out the following two lines to disable RAG
rag:
  enable_subagent: false  # set true to enable LLM post-filtering of RAG results
```

### How RAG Works

When RAG is enabled, two additional tools become available to the LLM:

- **`rag_query`** — Search the knowledge base by topic (e.g., "HIP shared memory optimization")
- **`rag_optimize`** — Get optimization suggestions for a kernel type on a target GPU

The LLM decides when to call these tools. Results are retrieved from a local knowledge base index via the `rag-mcp` server (spawned as a subprocess on first use).

### RAG Architecture

```
Agent → ToolRuntime.dispatch → MCPToolBridge → rag-mcp server (stdio subprocess)
                                                    │
                                            HybridRetriever.search()
                                              ├─ Embedding (semantic)
                                              ├─ BM25 (keyword)
                                              ├─ RRF Fusion
                                              └─ BGE Reranker
```

### RAG Server Config

The RAG MCP server config is at `mcp_tools/rag-mcp/src/rag_mcp/config/rag_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `retrieval.embed_top_k` | 25 | Embedding retrieval candidates |
| `retrieval.bm25_top_k` | 25 | BM25 retrieval candidates |
| `retrieval.enable_bm25` | true | Enable BM25 dual-path recall |
| `retrieval.mcp_top_k` | 8 | Final results returned |
| `reranker.enable_reranker` | true | Enable BGE reranker |
| `fusion.semantic_weight` | 0.7 | Embedding weight in fusion |
| `fusion.bm25_weight` | 0.3 | BM25 weight in fusion |

Override with `RAG_MCP_CONFIG` env var or `~/.config/rag-mcp/config.yaml`.

## Project Structure

```
src/minisweagent/
├── agents/                        # Agent implementations
│   ├── default.py                 #   Core agent (tool calling + bash)
│   ├── interactive.py             #   REPL-style interactive agent
│   ├── interactive_textual.py     #   Textual TUI agent
│   └── subagent.py                #   RAG result post-filtering sub-agent
├── tools/                         # Tool runtime & implementations
│   ├── tools_runtime.py           #   ToolRuntime: dispatch + RAG MCP registration
│   ├── tools.json                 #   Tool schemas (bash, submit, str_replace_editor, rag_query, rag_optimize)
│   ├── bash_command.py            #   Bash command execution
│   ├── str_replace_editor.py      #   File editor (view, create, str_replace, insert)
│   ├── submit.py                  #   Task submission
│   └── mcp_bridge.py              #   Sync/async bridge for MCP server communication
├── models/                        # LLM model interfaces
│   ├── amd_llm.py                 #   AMD LLM Gateway router (Claude)
│   ├── amd_base.py                #   Base class for AMD models
│   ├── amd_claude.py              #   Claude backend via AMD Gateway
│   ├── litellm_model.py           #   LiteLLM (supports most providers)
│   ├── anthropic_model.py         #   Anthropic direct
│   ├── openrouter_model.py        #   OpenRouter
│   └── portkey_model.py           #   Portkey
├── environments/                  # Execution environments
│   ├── local.py                   #   Local subprocess
│   ├── docker.py                  #   Docker/Podman
│   └── singularity.py             #   Singularity/Apptainer
├── config/                        # YAML config files
│   ├── mini.yaml                  #   Default config (claude-opus-4.5, yolo, RAG enabled)
│   ├── mini_rag.yaml              #   RAG-enabled config with subagent
│   ├── default.yaml               #   Base DefaultAgent config
│   └── ...
└── run/                           # CLI entry points
    ├── mini.py                    #   Main CLI (`mini` command)
    └── ...

mcp_tools/
├── mcp-client/                    # Generic MCP client (JSON-RPC over stdio)
│   └── src/mcp_client/
│       ├── client.py              #   MCPClient: subprocess management + protocol
│       ├── transport.py           #   Stdio transport layer
│       └── config.py              #   Server registry
└── rag-mcp/                       # RAG MCP server
    └── src/rag_mcp/
        ├── server.py              #   FastMCP server (query + optimize tools)
        ├── retrieval.py           #   HybridRetriever (embedding + BM25 + RRF + reranker)
        └── config/rag_config.yaml #   Retrieval config
```

## Configuration

Use `mini -c <config_name_or_path>` to select a config. Built-in configs:

| Config | Model | Mode | RAG | Description |
|--------|-------|------|-----|-------------|
| `mini.yaml` (default) | claude-opus-4.5 | yolo | enabled | Daily use, tool calling enabled |
| `mini_rag.yaml` | claude-opus-4.5 | yolo | enabled + subagent | RAG with LLM post-filtering |
| `default.yaml` | (not bound) | confirm | — | Generic base config |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `AMD_LLM_API_KEY` or `LLM_GATEWAY_KEY` | API key for AMD LLM Gateway |
| `RAG_MCP_CONFIG` | Override RAG server config path |
| `RAG_INDEX_PATH` | Override knowledge base index path |
