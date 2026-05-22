import json
import sys
from pathlib import Path
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.llm import make_llm
from common.console import preview, section, step


SALES_ROWS = [
    {"region": "South", "revenue": 120000, "churn_risk": "low"},
    {"region": "West", "revenue": 90000, "churn_risk": "high"},
    {"region": "North", "revenue": 150000, "churn_risk": "medium"},
]


class MultiAgentState(TypedDict):
    task: str
    agents: list[str]
    outputs: list[str]
    response: str


def parse_json(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def supervisor(state: MultiAgentState) -> dict:
    llm = make_llm(temperature=0)
    prompt = f"""
You are a multi-agent supervisor.

Available agents:
- data_analyst: analyzes the provided SALES_ROWS only
- web_researcher: provides external market context
- writer: writes the final summary

Task: {state["task"]}

Choose the minimum agents needed. Return only JSON:
{{"agents":["data_analyst","writer"]}}
"""
    data = parse_json(llm.invoke(prompt).content)
    step("supervisor_selected_agents", data["agents"])
    return {"agents": data["agents"]}


def data_analyst(state: MultiAgentState) -> dict:
    llm = make_llm(temperature=0)
    prompt = f"Analyze these sales rows for revenue and churn risk:\n{SALES_ROWS}"
    response = llm.invoke(prompt).content
    step("data_analyst_handoff", preview(response))
    return {"outputs": state["outputs"] + [f"data_analyst: {response}"]}


def web_researcher(state: MultiAgentState) -> dict:
    llm = make_llm(temperature=0)
    prompt = "Give one short generic market context note for a sales summary."
    response = llm.invoke(prompt).content
    step("web_researcher_handoff", preview(response))
    return {"outputs": state["outputs"] + [f"web_researcher: {response}"]}


def route_after_supervisor(state: MultiAgentState) -> str:
    return "data_analyst"


def route_after_analyst(state: MultiAgentState) -> str:
    return "web_researcher" if "web_researcher" in state["agents"] else "writer"


def writer(state: MultiAgentState) -> dict:
    llm = make_llm(temperature=0)
    prompt = f"""
Write a concise executive summary.

Task: {state["task"]}
Agent outputs:
{chr(10).join(state["outputs"])}
"""
    response = llm.invoke(prompt).content
    step("writer_finalized", preview(response))
    return {"response": response}


def build_graph():
    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("data_analyst", data_analyst)
    graph.add_node("web_researcher", web_researcher)
    graph.add_node("writer", writer)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor, ["data_analyst"])
    graph.add_conditional_edges("data_analyst", route_after_analyst, ["web_researcher", "writer"])
    graph.add_edge("web_researcher", "writer")
    graph.add_edge("writer", END)
    return graph.compile()


def run_agent(task: str) -> dict:
    app = build_graph()
    state = app.invoke({"task": task, "agents": [], "outputs": [], "response": ""})
    return {"state": state}


if __name__ == "__main__":
    section("Exercise 4 - Real LangGraph Multi-Agent System: Sales Brief")
    prompt = "Analyze the sales rows and summarize revenue and churn risk using only the provided data."
    result = run_agent(prompt)
    section(f"USER: {prompt}")
    step("agents", result["state"]["agents"])
    step("handoffs", [preview(output) for output in result["state"]["outputs"]])
    step("final_answer", result["state"]["response"])
    step("learning_goal", "Add evals for delegation, role adherence, handoff quality, grounded synthesis, and cost/waste.")
