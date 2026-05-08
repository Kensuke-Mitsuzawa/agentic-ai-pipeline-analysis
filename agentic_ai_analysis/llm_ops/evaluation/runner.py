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

    This is intentionally safe-by-default:
    - If sampling_rate == 0.0 -> no evaluation work is performed.
    - Metric frameworks are optional; this function can return empty results.
    """
    results: dict[str, List[MetricResult]] = {}
    for o in outcomes:
        if not should_evaluate(query_id=o.query_id, sampling_rate=sampling_rate):
            continue

        # Placeholder: frameworks will be integrated in later TODOs.
        # For now, return an empty metric set but keep the wiring stable.
        results[o.query_id] = []

    return results

