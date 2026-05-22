def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def step(label: str, value) -> None:
    print(f"[{label}] {value}")


def preview(text: str, limit: int = 500) -> str:
    text = str(text).replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."
