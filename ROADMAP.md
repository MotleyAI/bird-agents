# BIRD-Interact Agents — Multi-Framework Roadmap

Pluggable agent implementations for the [BIRD-Interact](https://github.com/bird-bench/BIRD-Interact) mini benchmark, evaluating [SLayer](https://github.com/MotleyAI/slayer) and raw SQL query generation.

## Architecture

```
bird-interact-agents/
└── agents/
    ├── claude_sdk/      ← Phase 1 (active)
    ├── pydantic_ai/     ← Phase 2
    ├── smolagents/      ← Phase 3
    ├── agno/            ← Phase 4
    └── mcp_agent/       ← Phase 5
```

All agents share the same evaluation harness from the BIRD-Interact repo (user simulator, action execution, test case evaluation). Each agent only implements the "brain" — the component that decides what tool to call given the current state.

Two **query modes**:
- **SLayer**: Agent interacts through SLayer's semantic layer (MCP tools). Doesn't know about SQL.
- **Raw**: Agent writes SQL directly using schema/knowledge exploration tools.

## Frameworks

| Phase | Framework | Package | Key Trait | Multi-Model | MCP |
|-------|-----------|---------|-----------|-------------|-----|
| 1 | **Claude Agent SDK** | `claude-agent-sdk` | Deepest tool loop (same engine as Claude Code) | Claude only | Native in-process MCP server |
| 2 | **PydanticAI** | `pydantic-ai` | Type-safe tools, `RunContext[Deps]`, Capabilities primitive | Any LiteLLM provider | Native (`MCPServerStdio`, toolsets) |
| 3 | **smolagents** | `smolagents` | Code-as-action (agents write Python), HuggingFace ecosystem | LiteLLMModel | `Tool.from_mcp()` |
| 4 | **Agno** | `agno` | Context engineering, 100+ built-in toolkits, multi-agent Teams | Yes | MCP streaming |
| 5 | **mcp-agent** | `mcp-agent` | Full MCP spec (sampling, elicitation, roots), Temporal durability | Yes | Full spec |

### Phase 1: Claude Agent SDK

The deepest integration with Claude's tool-use capabilities. Tools are defined with `@tool` decorators and served via an in-process MCP server. The SDK manages the full agent loop (tool calls → results → next step) automatically.

**Trade-off**: Claude-locked. Cannot benchmark against GPT, Gemini, or open models.

### Phase 2: PydanticAI

The strongest open alternative. Tools are defined with `@agent.tool` decorators and get full Pydantic validation on inputs/outputs. Dependencies are injected via `RunContext[Deps]`. The Capabilities primitive (v1.71+) lets you bundle SLayer tools + prompts + model settings into a reusable package.

**Key advantage**: Multi-model benchmarking (any LiteLLM-compatible provider).

### Phase 3: smolagents

HuggingFace's framework where agents write Python code as actions rather than making JSON tool calls. This is more powerful for multi-step reasoning — the agent can compose tool calls in a single code block.

### Phase 4: Agno

Context engineering-focused framework (formerly Phidata). 100+ built-in toolkits. Multi-agent Teams for collaborative workflows.

### Phase 5: mcp-agent

The most MCP-faithful option. Implements the full MCP spec including sampling, elicitation, and roots. One config line gets durable execution via Temporal.

## SLayer Integration

For SLayer mode across all frameworks:
1. **One-time setup**: Auto-ingest mini-interact SQLite DBs into SLayer datasources + models
2. **Agent tools**: SLayer MCP tools (`help`, `list_datasources`, `models_summary`, `inspect_model`, `query`)
3. **Submission**: Agent submits SLayer query → SLayer generates SQL → SQL evaluated against ground truth
