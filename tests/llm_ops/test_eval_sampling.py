from agentic_ai_analysis.llm_ops.evaluation.sampling import should_evaluate


def test_should_evaluate_deterministic():
    qid = "abc123"
    a = should_evaluate(query_id=qid, sampling_rate=0.25)
    b = should_evaluate(query_id=qid, sampling_rate=0.25)
    assert a == b


def test_should_evaluate_rate_bounds():
    assert should_evaluate(query_id="x", sampling_rate=0.0) is False
    assert should_evaluate(query_id="x", sampling_rate=1.0) is True

