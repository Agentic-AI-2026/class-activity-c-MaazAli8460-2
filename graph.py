### Graph
from __future__ import annotations

import ast
import importlib
import json
import math
import os
import re
from typing import Any, Literal, NotRequired, TypedDict

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph


def _try_load_dotenv() -> None:
	"""Load .env values if python-dotenv is installed."""
	try:
		load_dotenv = getattr(importlib.import_module("dotenv"), "load_dotenv")
		load_dotenv()
	except Exception:
		return


_try_load_dotenv()


class Step(TypedDict, total=False):
	thought: str
	action: str
	action_input: Any
	observation: str


class AgentState(TypedDict, total=False):
	input: str
	agent_scratchpad: str
	final_answer: str
	steps: list[Step]
	action: str
	action_input: Any
	iteration_count: int
	max_iterations: int


CITY_COORDS = {
	"london": (51.5074, -0.1278),
	"paris": (48.8566, 2.3522),
	"new york": (40.7128, -74.0060),
	"tokyo": (35.6762, 139.6503),
	"karachi": (24.8607, 67.0011),
	"lahore": (31.5204, 74.3587),
	"islamabad": (33.6844, 73.0479),
	"rawalpindi": (33.5651, 73.0169),
	"dubai": (25.2048, 55.2708),
	"berlin": (52.5200, 13.4050),
	"sydney": (-33.8688, 151.2093),
	"chicago": (41.8781, -87.6298),
}


@tool
def calculator(expression: str) -> str:
	"""Evaluate a math expression safely (supports +, -, *, /, **, sqrt, log, sin, cos, pi, e)."""
	safe_globals = {
		"__builtins__": {},
		"sqrt": math.sqrt,
		"log": math.log,
		"log2": math.log2,
		"log10": math.log10,
		"sin": math.sin,
		"cos": math.cos,
		"tan": math.tan,
		"ceil": math.ceil,
		"floor": math.floor,
		"pi": math.pi,
		"e": math.e,
		"abs": abs,
		"round": round,
		"pow": pow,
	}
	try:
		value = eval(expression, safe_globals)
		return f"{expression} = {round(float(value), 6)}"
	except ZeroDivisionError:
		return "Error: Division by zero"
	except SyntaxError:
		return f"Error: Invalid syntax in '{expression}'"
	except Exception as exc:
		return f"Error evaluating '{expression}': {exc}"


@tool
def get_current_weather(city: str) -> str:
	"""Get current weather for a city (temperature, condition, wind, humidity, and feels-like)."""
	coords = CITY_COORDS.get(city.lower().strip())
	if not coords:
		available = ", ".join(name.title() for name in CITY_COORDS)
		return f"City '{city}' not found. Available cities: {available}"

	lat, lon = coords
	url = (
		"https://api.open-meteo.com/v1/forecast"
		f"?latitude={lat}&longitude={lon}"
		"&current_weather=true"
		"&hourly=relativehumidity_2m,apparent_temperature"
	)

	try:
		data = requests.get(url, timeout=8).json()
		cw = data.get("current_weather", {})
		temp = cw.get("temperature", "N/A")
		wind = cw.get("windspeed", "N/A")
		weather_code = cw.get("weathercode", 0)
		condition = "Sunny" if weather_code < 3 else "Cloudy" if weather_code < 50 else "Rainy"
		humidity = data.get("hourly", {}).get("relativehumidity_2m", ["N/A"])[0]
		feels_like = data.get("hourly", {}).get("apparent_temperature", ["N/A"])[0]

		return (
			f"Current weather in {city.title()}:\n"
			f"  Condition : {condition}\n"
			f"  Temp      : {temp} C\n"
			f"  Feels like: {feels_like} C\n"
			f"  Wind      : {wind} km/h\n"
			f"  Humidity  : {humidity}%"
		)
	except requests.Timeout:
		return f"Weather API timed out for '{city}'"
	except Exception as exc:
		return f"Weather API error: {exc}"


def _strip_html(text: str) -> str:
	return re.sub(r"<[^>]+>", "", text or "")


@tool
def search_web(query: str) -> str:
	"""Search factual information on the web (encyclopedia-style results)."""
	api_key = os.getenv("TAVILY_API_KEY", "").strip()
	if api_key:
		try:
			TavilyClient = getattr(importlib.import_module("tavily"), "TavilyClient")
			tavily = TavilyClient(api_key=api_key)
			response = tavily.search(query=query, search_depth="basic", max_results=3)
			results = response.get("results", [])
			if results:
				return "\n\n".join(
					[
						f"[{idx + 1}] {item.get('title', 'Untitled')}\n"
						f"    {item.get('content', '').strip()}\n"
						f"    Source: {item.get('url', 'Unknown')}"
						for idx, item in enumerate(results)
					]
				)
		except Exception:
			# Fall back to Wikipedia-based search if Tavily is unavailable.
			pass

	params = {
		"action": "query",
		"list": "search",
		"srsearch": query,
		"format": "json",
		"utf8": 1,
		"srlimit": 3,
	}
	try:
		response = requests.get("https://en.wikipedia.org/w/api.php", params=params, timeout=8)
		response.raise_for_status()
		payload = response.json()
		results = payload.get("query", {}).get("search", [])
		if not results:
			return f"No results found for '{query}'."

		lines = []
		for idx, item in enumerate(results, start=1):
			title = item.get("title", "Untitled")
			snippet = _strip_html(item.get("snippet", "")).replace("\n", " ").strip()
			url_title = title.replace(" ", "_")
			lines.append(
				f"[{idx}] {title}\n"
				f"    {snippet}\n"
				f"    Source: https://en.wikipedia.org/wiki/{url_title}"
			)
		return "\n\n".join(lines)
	except requests.Timeout:
		return f"Search timed out for '{query}'."
	except Exception as exc:
		return f"Search error: {exc}"


TOOLS: list[BaseTool] = [calculator, get_current_weather, search_web]
TOOL_MAP: dict[str, BaseTool] = {tool_obj.name: tool_obj for tool_obj in TOOLS}


REACT_SYSTEM_PROMPT = """You are a ReAct agent.
Use this loop exactly: Thought -> Action -> Observation -> Thought -> ...

Available tools:
{tool_descriptions}

Rules:
1. Use tools for factual information and calculations.
2. Use one tool call at a time.
3. When ready, provide a final response.
4. Never output both Action and Final Answer in one response.
5. If a tool returns an error or no useful new information twice, stop looping and provide the best possible Final Answer from available observations.
6. For multi-part questions, collect evidence for each part before finalizing.

Output format:
Thought: <your brief reasoning>
Action: <tool name>
Action Input: <valid JSON object>

OR

Thought: <your brief reasoning>
Final Answer: <answer for user>
"""


THOUGHT_RE = re.compile(
	r"Thought\s*:\s*(.+?)(?=\n\s*(?:Action|Final Answer)\s*:|$)",
	flags=re.IGNORECASE | re.DOTALL,
)
ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", flags=re.IGNORECASE)
ACTION_INPUT_RE = re.compile(r"Action Input\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)
FINAL_RE = re.compile(r"Final Answer\s*:\s*(.+)", flags=re.IGNORECASE | re.DOTALL)


class ParsedOutput(TypedDict):
	kind: Literal["action", "final"]
	thought: str
	action: NotRequired[str]
	action_input: NotRequired[Any]
	final_answer: NotRequired[str]


def _render_tool_descriptions() -> str:
	lines: list[str] = []
	for t in TOOLS:
		arg_names = ", ".join(t.args.keys()) if t.args else "no args"
		lines.append(f"- {t.name}({arg_names}): {t.description}")
	return "\n".join(lines)


def _parse_action_input(raw_text: str) -> Any:
	cleaned = raw_text.strip()
	try:
		return json.loads(cleaned)
	except Exception:
		pass

	try:
		return ast.literal_eval(cleaned)
	except Exception:
		return cleaned


def _parse_react_output(text: str) -> ParsedOutput:
	thought_match = THOUGHT_RE.search(text)
	thought = thought_match.group(1).strip() if thought_match else ""

	final_match = FINAL_RE.search(text)
	if final_match:
		return {
			"kind": "final",
			"thought": thought,
			"final_answer": final_match.group(1).strip(),
		}

	action_match = ACTION_RE.search(text)
	action_input_match = ACTION_INPUT_RE.search(text)
	if action_match:
		action_input: Any = {}
		if action_input_match:
			action_input = _parse_action_input(action_input_match.group(1))

		return {
			"kind": "action",
			"thought": thought,
			"action": action_match.group(1).strip(),
			"action_input": action_input,
		}

	return {
		"kind": "final",
		"thought": thought,
		"final_answer": text.strip(),
	}


def _append_scratchpad(existing: str, new_block: str) -> str:
	if not existing:
		return new_block
	return f"{existing}\n{new_block}"


def _tool_input_for_call(selected_tool: BaseTool, action_input: Any) -> Any:
	if isinstance(action_input, dict):
		return action_input

	if action_input is None:
		return {}

	arg_names = list(selected_tool.args.keys()) if selected_tool.args else []
	if len(arg_names) == 1:
		return {arg_names[0]: action_input}
	return action_input


def build_react_app(llm):
	"""Build and compile the LangGraph ReAct workflow."""

	tool_descriptions = _render_tool_descriptions()

	def react_node(state: AgentState) -> AgentState:
		current_iter = state.get("iteration_count", 0)
		max_iter = state.get("max_iterations", 12)
		if current_iter >= max_iter:
			return {
				"final_answer": "Stopped: maximum reasoning steps reached before final answer.",
				"iteration_count": current_iter,
			}

		messages = [
			SystemMessage(content=REACT_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)),
			HumanMessage(
				content=(
					f"Question: {state['input']}\n\n"
					f"Scratchpad:\n{state.get('agent_scratchpad', '')}\n\n"
					"Return only one ReAct step in the required format."
				)
			),
		]

		response = llm.invoke(messages)
		content = response.content if isinstance(response.content, str) else str(response.content)
		parsed = _parse_react_output(content)

		updated_steps = list(state.get("steps", []))

		if parsed["kind"] == "final":
			thought = parsed.get("thought", "")
			scratch_block = ""
			if thought:
				scratch_block += f"Thought: {thought}\n"
			scratch_block += f"Final Answer: {parsed['final_answer']}"

			return {
				"agent_scratchpad": _append_scratchpad(state.get("agent_scratchpad", ""), scratch_block),
				"final_answer": parsed["final_answer"],
				"iteration_count": current_iter + 1,
			}

		action = parsed["action"]
		action_input = parsed.get("action_input", {})
		thought = parsed.get("thought", "")

		updated_steps.append(
			{
				"thought": thought,
				"action": action,
				"action_input": action_input,
			}
		)

		try:
			action_input_text = json.dumps(action_input, ensure_ascii=True)
		except TypeError:
			action_input_text = str(action_input)

		scratch_block = (
			f"Thought: {thought}\n"
			f"Action: {action}\n"
			f"Action Input: {action_input_text}"
		)

		return {
			"action": action,
			"action_input": action_input,
			"steps": updated_steps,
			"agent_scratchpad": _append_scratchpad(state.get("agent_scratchpad", ""), scratch_block),
			"iteration_count": current_iter + 1,
		}

	def tool_node(state: AgentState) -> AgentState:
		action = state.get("action", "")
		action_input = state.get("action_input", {})
		selected_tool = TOOL_MAP.get(action)

		if not selected_tool:
			observation = f"Unknown tool '{action}'. Available tools: {', '.join(TOOL_MAP)}"
		else:
			try:
				prepared_input = _tool_input_for_call(selected_tool, action_input)
				result = selected_tool.invoke(prepared_input)
				observation = str(result)
			except Exception as exc:
				observation = f"Tool execution error in '{action}': {exc}"

		updated_steps = list(state.get("steps", []))
		if updated_steps:
			updated_steps[-1]["observation"] = observation

		scratch_block = f"Observation: {observation}"
		return {
			"steps": updated_steps,
			"agent_scratchpad": _append_scratchpad(state.get("agent_scratchpad", ""), scratch_block),
			"action": "",
			"action_input": {},
		}

	def route_after_react(state: AgentState) -> Literal["is_action", "is_final"]:
		if state.get("final_answer"):
			return "is_final"
		return "is_action" if state.get("action") else "is_final"

	workflow = StateGraph(AgentState)
	workflow.add_node("react_node", react_node)
	workflow.add_node("tool_node", tool_node)

	workflow.add_edge(START, "react_node")
	workflow.add_conditional_edges(
		"react_node",
		route_after_react,
		{
			"is_action": "tool_node",
			"is_final": END,
		},
	)
	workflow.add_edge("tool_node", "react_node")

	return workflow.compile()


def run_query(app, query: str, max_iterations: int = 20) -> AgentState:
	"""Run a query through the compiled graph and return final state."""
	initial_state: AgentState = {
		"input": query,
		"agent_scratchpad": "",
		"final_answer": "",
		"steps": [],
		"action": "",
		"action_input": {},
		"iteration_count": 0,
		"max_iterations": max_iterations,
	}
	return app.invoke(initial_state, config={"recursion_limit": max_iterations * 4})
