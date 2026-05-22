# Multi-Agent Eval Notes

Add evals for:

- Delegation: correct agents selected for the task.
- Forbidden agents: avoid external research for local-data-only tasks.
- Role adherence: analyst should compute, writer should synthesize, researcher should not invent.
- Handoff quality: intermediate outputs should contain useful facts.
- Final synthesis: final answer should be grounded in agent outputs.
- Cost/latency: avoid unnecessary agents.

Suggested first cases:

1. CSV-only analysis should use `data_analyst` and `writer`, not `web_researcher`.
2. A market-research task may use `web_researcher`.
3. Final answer should mention total revenue and high West churn risk.

