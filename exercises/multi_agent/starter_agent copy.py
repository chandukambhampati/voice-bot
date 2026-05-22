import json
import sys
from pathlib import Path
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langchain.agents import create_agent

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.llm import make_llm
from common.console import preview, section, step


SALES_ROWS = [
    {"region": "South", "revenue": 120000, "churn_risk": "low"},
    {"region": "West",  "revenue": 90000,  "churn_risk": "high"},
    {"region": "North", "revenue": 150000, "churn_risk": "medium"},
]


# ---------------------------------------------------------------------------
# Shared parent state
# ---------------------------------------------------------------------------

class MultiAgentState(TypedDict):
    task: str
    agents: list[str]
    outputs: list[str]
    response: str


# ---------------------------------------------------------------------------
# Helper — extract the final AI text from a create_react_agent response
# ---------------------------------------------------------------------------

def _last_ai_content(agent_result: dict) -> str:
    for msg in reversed(agent_result["messages"]):
        if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
            return msg.content
    return ""


# ===========================================================================
# PATTERN 1 — create_react_agent
# data_analyst and writer are ReAcT agents with tools
# ===========================================================================

# --- data_analyst tools ---

@tool
def get_sales_data() -> str:
    """Fetches the raw sales rows (region, revenue, churn_risk)."""
    return json.dumps(SALES_ROWS, indent=2)


@tool
def calculate_revenue_summary() -> str:
    """Calculates total revenue and a per-region revenue breakdown."""
    total = sum(row["revenue"] for row in SALES_ROWS)
    by_region = {row["region"]: row["revenue"] for row in SALES_ROWS}
    return json.dumps({"total_revenue": total, "by_region": by_region})


@tool
def get_churn_risk_breakdown() -> str:
    """Returns the churn risk level for every region."""
    return json.dumps({row["region"]: row["churn_risk"] for row in SALES_ROWS})


def data_analyst(state: MultiAgentState) -> dict:
    """ReAcT agent: fetches and analyzes sales data using tools."""
    agent = create_agent(
        model=make_llm(temperature=0),
        tools=[get_sales_data, calculate_revenue_summary, get_churn_risk_breakdown],
        system_prompt=SystemMessage(content=(
            "You are a senior data analyst. "
            "Use your tools to fetch and analyze the sales data. "
            "Report on revenue totals and churn risk patterns. "
            "Be concise and structured. Do NOT write the final summary."
        )),
    )
    result = agent.invoke({"messages": [HumanMessage(content=state["task"])]})
    output = _last_ai_content(result)
    step("data_analyst_handoff", preview(output))
    return {"outputs": state["outputs"] + [f"[data_analyst]: {output}"]}


# --- writer tools ---

@tool
def format_executive_summary(title: str, sections: str) -> str:
    """
    Formats a polished executive summary.

    Args:
        title:    The report headline.
        sections: A JSON-encoded list of {'heading': str, 'body': str} dicts.
    """
    parsed = json.loads(sections)
    lines = [f"# {title}", ""]
    for sec in parsed:
        lines.append(f"## {sec['heading']}")
        lines.append(sec["body"])
        lines.append("")
    return "\n".join(lines)


def writer(state: MultiAgentState) -> dict:
    """ReAcT agent: synthesizes prior outputs into a formatted executive summary."""
    prior = "\n\n".join(state["outputs"]) or "None."
    agent = create_agent(
        model=make_llm(temperature=0),
        tools=[format_executive_summary],
        system_prompt=SystemMessage(content=(
            "You are an executive communications writer. "
            "Use the format_executive_summary tool to produce a polished report. "
            "Base everything strictly on the prior agent outputs — do NOT invent figures."
        )),
    )
    result = agent.invoke({
        "messages": [HumanMessage(content=(
            f"Task: {state['task']}\n\nAgent outputs to synthesize:\n{prior}"
        ))]
    })
    output = _last_ai_content(result)
    step("writer_finalized", preview(output))
    return {
        "outputs": state["outputs"] + [f"[writer]: {output}"],
        "response": output,
    }


# ===========================================================================
# PATTERN 2 — Sub-graph
# web_researcher is a compiled sub-graph with two internal nodes:
#   fetch_context  → ReAcT agent that calls get_market_context tool
#   validate_context → plain LLM that critiques and refines the fetched context
# ===========================================================================

class ResearchState(TypedDict):
    """
    Sub-graph state. Uses the same key names as the parent so that
    LangGraph automatically passes 'task' and 'outputs' in and back out.
    """
    task: str
    outputs: list[str]


# --- web_researcher sub-graph tool ---

@tool
def get_market_context(topic: str) -> str:
    """
    Returns a concise external market context paragraph for the given topic.
    Topics: 'churn', 'revenue', or any general sales term.
    """
    contexts = {
        "churn": (
            "Industry benchmarks show B2B SaaS churn rates averaging 5-7% annually. "
            "High-churn regions often correlate with competitive pricing pressure and "
            "slower enterprise adoption cycles."
        ),
        "revenue": (
            "Global B2B software revenue grew ~12% YoY in 2024, driven by cloud migration "
            "and AI adoption. Regional disparities persist, with western markets outpacing "
            "emerging segments in deal size."
        ),
        "default": (
            "Current market conditions show moderate growth with increasing pressure on "
            "customer retention. Organisations prioritising customer success teams report "
            "up to 30% lower churn rates than peers."
        ),
    }
    key = next((k for k in contexts if k in topic.lower()), "default")
    return contexts[key]


# --- Sub-graph node 1: fetch_context ---

def fetch_context(state: ResearchState) -> dict:
    """
    ReAcT agent inside the sub-graph.
    Calls get_market_context tool to retrieve external market data.
    """
    agent = create_agent(
        model=make_llm(temperature=0),
        tools=[get_market_context],
        system_prompt=SystemMessage(content=(
            "You are a market data fetcher. "
            "Call the get_market_context tool with the most relevant topic "
            "(churn, revenue, or a related term) for the task. "
            "Return the raw fetched context — do not summarize yet."
        )),
    )
    result = agent.invoke({"messages": [HumanMessage(content=state["task"])]})
    raw_context = _last_ai_content(result)
    step("fetch_context_raw", preview(raw_context))
    # Store temporarily in outputs so validate_context can read it
    return {"outputs": state["outputs"] + [f"[fetch_context]: {raw_context}"]}


# --- Sub-graph node 2: validate_context ---

def validate_context(state: ResearchState) -> dict:
    """
    Plain LLM node inside the sub-graph.
    Reviews the fetched context, removes irrelevant parts, and emits
    a clean handoff as [web_researcher]: ...
    """
    llm = make_llm(temperature=0)
    fetched = next(
        (o for o in reversed(state["outputs"]) if o.startswith("[fetch_context]")),
        "No context fetched.",
    )
    messages = [
        SystemMessage(content=(
            "You are a research quality analyst. "
            "Review the fetched market context below and refine it into one "
            "clear, relevant paragraph for a sales executive summary. "
            "Remove anything generic or off-topic. "
            "Do NOT repeat the sales data. Do NOT write the final summary."
        )),
        HumanMessage(content=(
            f"Task: {state['task']}\n\n"
            f"Fetched context:\n{fetched}"
        )),
    ]
    refined = llm.invoke(messages).content
    step("web_researcher_validated", preview(refined))

    # Replace the raw fetch_context entry with the clean web_researcher handoff
    clean_outputs = [o for o in state["outputs"] if not o.startswith("[fetch_context]")]
    return {"outputs": clean_outputs + [f"[web_researcher]: {refined}"]}


# --- Build and compile the sub-graph ---

def build_research_subgraph():
    sg = StateGraph(ResearchState)
    sg.add_node("fetch_context",    fetch_context)
    sg.add_node("validate_context", validate_context)
    sg.add_edge(START, "fetch_context")
    sg.add_edge("fetch_context", "validate_context")
    sg.add_edge("validate_context", END)
    return sg.compile()


research_subgraph = build_research_subgraph()


# ===========================================================================
# Supervisor — plain LLM; decides which agents/subgraphs to invoke
# ===========================================================================

def supervisor(state: MultiAgentState) -> dict:
    llm = make_llm(temperature=0)
    messages = [
        SystemMessage(content=(
            "You are a multi-agent supervisor.\n"
            "Available agents:\n"
            "  - data_analyst      : ReAcT agent — analyzes sales rows with tools\n"
            "  - web_researcher    : Sub-graph  — fetches and validates market context\n"
            "  - writer            : ReAcT agent — formats the final executive summary\n\n"
            "Rules:\n"
            "  1. Always include 'data_analyst' and 'writer'.\n"
            "  2. Include 'web_researcher' only if external market context is explicitly needed.\n"
            "  3. Return ONLY valid JSON: {\"agents\": [\"data_analyst\", \"writer\"]}"
        )),
        HumanMessage(content=f"Task: {state['task']}"),
    ]
    raw = llm.invoke(messages).content
    cleaned = raw.strip().replace("```json", "").replace("```", "").strip()
    agents = json.loads(cleaned)["agents"]
    step("supervisor_selected_agents", agents)
    return {"agents": agents}


# ===========================================================================
# Routing
# ===========================================================================

def route_after_supervisor(state: MultiAgentState) -> str:
    return "data_analyst"


def route_after_analyst(state: MultiAgentState) -> str:
    # Route to the sub-graph if selected, otherwise straight to writer
    return "web_researcher" if "web_researcher" in state["agents"] else "writer"


# ===========================================================================
# Parent graph — combines create_react_agent nodes + sub-graph node
# ===========================================================================

def build_graph():
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor",     supervisor)
    graph.add_node("data_analyst",   data_analyst)       # create_react_agent
    graph.add_node("web_researcher", research_subgraph)  # compiled sub-graph
    graph.add_node("writer",         writer)             # create_react_agent

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor",   route_after_supervisor, ["data_analyst"])
    graph.add_conditional_edges("data_analyst", route_after_analyst,    ["web_researcher", "writer"])
    graph.add_edge("web_researcher", "writer")
    graph.add_edge("writer", END)

    return graph.compile()


def run_agent(task: str) -> dict:
    app = build_graph()
    state = app.invoke({"task": task, "agents": [], "outputs": [], "response": ""})
    return {"state": state}


if __name__ == "__main__":
    section("Exercise 4 - Real LangGraph Multi-Agent System: Sales Brief")
    prompt = (
        "Analyze the sales rows and summarize revenue and churn risk. "
        "Also include relevant external market context."
    )
    result = run_agent(prompt)
    section(f"USER: {prompt}")
    step("agents",       result["state"]["agents"])
    step("handoffs",     [preview(o) for o in result["state"]["outputs"]])
    step("final_answer", result["state"]["response"])
    step("learning_goal", "Add evals for delegation, role adherence, handoff quality, grounded synthesis, and cost/waste.")

    graph_image_path = Path(__file__).parent / "stategraph.png"
    with open(graph_image_path, "wb") as f:
        f.write(build_graph().get_graph().draw_mermaid_png())
    step("graph_saved", f"Mermaid PNG saved to {graph_image_path}")
