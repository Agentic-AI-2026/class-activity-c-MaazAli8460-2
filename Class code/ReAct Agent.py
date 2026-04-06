# ─── ReAct AGENT with MCP Tools ──────────────────────────────────────────────
REACT_SYSTEM = """You are a ReAct agent. Strictly follow this loop:
Thought → Action (tool call) → Observation → Thought → ...

RULES:
1. ALWAYS use a tool for factual information — never answer from memory expect the founding years of companies.
2. For multi-part questions, make one tool call per fact.
3. ALWAYS use calculator for any arithmetic — never compute in your head.
4. Only give Final Answer AFTER all required tool calls are complete."""

async def react_agent_mcp(user_input: str, max_steps: int = 20) -> str:
    """ReAct loop using real MCP math servers."""
    tools, tools_map = await get_mcp_tools(["math"])
    llm_react = llm.bind_tools(tools)
    messages = [
        SystemMessage(content=REACT_SYSTEM),
        HumanMessage(content=user_input)
    ]
    for step in range(max_steps):
        response = llm_react.invoke(messages)
        messages.append(response)        
        if not response.tool_calls:
            print(f"\n Final Answer (step {step+1}): {response.content}")
            return response.content
        for tc in response.tool_calls:
            print(f"   Step {step+1} | [{tc['name']}] via MCP | Args: {tc['args']}")
            # MCP tool call — async
            result = await tools_map[tc["name"]].ainvoke(tc["args"])
            print(f"      Observation: {str(result)}")
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    return " Max steps reached."
