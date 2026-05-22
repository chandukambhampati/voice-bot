import os
from pathlib import Path

from langchain_openai import ChatOpenAI


DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_LANGSMITH_PROJECT = "mission8-agent-evals"


def load_mission_env() -> None:
    # Try local directory first (root), then parents[2] for sub-exercise folder layouts
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            env_path = Path.cwd() / ".env"
            if not env_path.exists():
                return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


def require_openai_key() -> None:
    load_mission_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it before running this real-agent lab."
        )
    configure_observability()


def configure_observability() -> None:
    load_mission_env()
    os.environ.setdefault("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT)
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", DEFAULT_LANGSMITH_PROJECT))

    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

    # Keep demo runs clean by tracing only when explicitly enabled.
    # Set LANGSMITH_TRACING=true in mission8/.env after confirming the key works.
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def model_name() -> str:
    load_mission_env()
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def make_llm(temperature: float = 0) -> ChatOpenAI:
    require_openai_key()
    configure_observability()
    return ChatOpenAI(
        model=model_name(),
        temperature=temperature,
    )
