import os

from agentic_ai_analysis.llm_ops.langfuse_tracing import get_tracer


def test_langfuse_noop_without_config(monkeypatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.setenv("LANGFUSE_ENABLED", "0")
    monkeypatch.setenv("LANGFUSE_TRACER_RESET", "1")

    tracer = get_tracer()
    trace = tracer.start_trace(trace_id="qid", name="test", input="hello", metadata={})
    with tracer.span(trace, name="span1", input="in"):
        pass
    tracer.score(trace, name="m", value=1.0)

