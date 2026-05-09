from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationRecord(BaseModel):
    query_id: str
    question: str
    answer: str
    contexts: list[str] = Field(default_factory=list)


class MetricResult(BaseModel):
    name: str
    value: float
    details: dict = Field(default_factory=dict)

