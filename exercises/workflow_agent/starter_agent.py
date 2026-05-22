import json
import sys
from pathlib import Path
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.llm import make_llm
from common.console import section, step


class WorkflowState(TypedDict):
    request: str
    intent: str
    risk: str
    approval_required: bool
    action: str
    response: str
    nodes: list[str]


def parse_json(text: str) -> dict:
    cleaned = text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(cleaned)


def normalize_intent(raw_intent: str, request: str) -> str:
    lower = request.lower()
    if any(word in lower for word in ("admin", "production", "delete", "destructive")):
        return "high_risk_access"
    allowed = {"standard_support", "high_risk_access", "general"}
    if raw_intent in allowed:
        return raw_intent

    # LLMs occasionally echo the schema. Keep the exercise robust and visible.
    if "reset" in lower or "password" in lower:
        return "standard_support"
    return "general"


def classify_request(state: WorkflowState) -> dict:
    llm = make_llm(temperature=0)
    prompt = f"""
Classify this IT request.

Request: {state["request"]}

Return only JSON with one selected value:
{{"intent":"standard_support"}}
"""
    data = parse_json(llm.invoke(prompt).content)
    data["intent"] = normalize_intent(data.get("intent", ""), state["request"])
    step("classify_request", data)
    return {"intent": data["intent"], "nodes": state["nodes"] + ["classify_request"]}


def check_policy(state: WorkflowState) -> dict:
    llm = make_llm(temperature=0)
    prompt = f"""
Assess risk for this IT request.

Request: {state["request"]}
Intent: {state["intent"]}

High-risk access includes admin access, production access, account deletion, or destructive changes.

Return only JSON:
{{"risk":"low|high","approval_required":true_or_false}}
"""
    data = parse_json(llm.invoke(prompt).content)
    step("check_policy", data)
    return {
        "risk": data["risk"],
        "approval_required": bool(data["approval_required"]),
        "nodes": state["nodes"] + ["check_policy"],
    }


def route_after_policy(state: WorkflowState) -> str:
    return "request_approval" if state["approval_required"] else "execute_action"


def request_approval(state: WorkflowState) -> dict:
    step("route", "request_approval")
    return {"action": "waiting_for_approval", "nodes": state["nodes"] + ["request_approval"]}


def execute_action(state: WorkflowState) -> dict:
    step("route", "execute_action")
    return {"action": "executed", "nodes": state["nodes"] + ["execute_action"]}


def final_response(state: WorkflowState) -> dict:
    llm = make_llm(temperature=0)
    prompt = f"""
Write a short IT helpdesk response.

Request: {state["request"]}
Risk: {state["risk"]}
Approval required: {state["approval_required"]}
Action: {state["action"]}
"""
    return {"response": llm.invoke(prompt).content, "nodes": state["nodes"] + ["final_response"]}


def build_graph():
    graph = StateGraph(WorkflowState)
    graph.add_node("classify_request", classify_request)
    graph.add_node("check_policy", check_policy)
    graph.add_node("request_approval", request_approval)
    graph.add_node("execute_action", execute_action)
    graph.add_node("final_response", final_response)
    graph.add_edge(START, "classify_request")
    graph.add_edge("classify_request", "check_policy")
    graph.add_conditional_edges(
        "check_policy",
        route_after_policy,
        ["request_approval", "execute_action"],
    )
    graph.add_edge("request_approval", "final_response")
    graph.add_edge("execute_action", "final_response")
    graph.add_edge("final_response", END)
    return graph.compile()


def run_agent(request: str) -> dict:
    app = build_graph()
    state = app.invoke(
        {
            "request": request,
            "intent": "",
            "risk": "",
            "approval_required": False,
            "action": "",
            "response": "",
            "nodes": [],
        }
    )
    return {"state": state}


if __name__ == "__main__":
    section("Exercise 3 - Real LangGraph Workflow: IT Access Request")
    prompts = ["Please reset my password.", "Give me admin access to production."]
    for prompt in prompts:
        section(f"USER: {prompt}")
        result = run_agent(prompt)
        step("visited_nodes", result["state"]["nodes"])
        step("final_state", {k: result["state"][k] for k in ["intent", "risk", "approval_required", "action"]})
        step("final_answer", result["state"]["response"])
        step("learning_goal", "Add evals for route, node sequence, state transitions, approval gates, and forbidden nodes.")
