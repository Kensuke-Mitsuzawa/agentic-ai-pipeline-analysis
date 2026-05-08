from __future__ import annotations

from typing import Iterable, List

from agentic_ai_analysis.agents.data_models import PipelineOutcome

from .models import EvaluationRecord, MetricResult
from .sampling import should_evaluate


def build_record(outcome: PipelineOutcome) -> EvaluationRecord:
    # Minimal extraction from the existing pipeline.
    # - question: outcome.prompt
    # - answer: outcome.final_outcome
    # - contexts: use judge explanations if present (best-effort)
    contexts: list[str] = []
    node = outcome.nodes.get("agent_4_judge_docs")
    if node is not None:
        try:
            arg = getattr(node, "args", {}) or {}
            expl = arg.get("explanation")
            if isinstance(expl, str) and expl.strip():
                contexts.append(expl.strip())
        except Exception:
            pass

    return EvaluationRecord(
        query_id=outcome.query_id,
        question=outcome.prompt,
        answer=outcome.final_outcome,
        contexts=contexts,
    )


def evaluate_outcomes(
    outcomes: Iterable[PipelineOutcome],
    *,
    sampling_rate: float = 0.0,
) -> dict[str, List[MetricResult]]:
    """
    Sample-and-evaluate pipeline outcomes.
    """
    from agentic_ai_analysis.agents.data_models import PromptContext

    results: dict[str, List[MetricResult]] = {}
    for o in outcomes:
        if not should_evaluate(query_id=o.query_id, sampling_rate=sampling_rate):
            continue

        metrics: List[MetricResult] = []

        # 1. Answer Presence & Length
        answer = o.final_outcome or ""
        metrics.append(MetricResult(name="answer_length", value=float(len(answer))))
        metrics.append(MetricResult(name="has_answer", value=1.0 if len(answer.strip()) > 0 else 0.0))

        # 2. Correctness (if options are available in prompt JSON)
        try:
            ctx = PromptContext.model_validate_json(o.prompt)
            if ctx.options:
                # SciQ logic: first option is the correct one
                ground_truth = ctx.options[0].lower().strip()
                is_correct = 1.0 if ground_truth in answer.lower() else 0.0
                metrics.append(MetricResult(name="is_correct", value=is_correct))
        except Exception:
            pass

        # 3. Retrieval Relevance (from Judge agent)
        judge_node = o.nodes.get("agent_4_judge_docs")
        if judge_node:
            is_related = judge_node.args.get("is_related", False)
            metrics.append(MetricResult(name="context_relevance", value=1.0 if is_related else 0.0))

        results[o.query_id] = metrics

    return results

