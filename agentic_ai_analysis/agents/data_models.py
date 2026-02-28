from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union

class BaseNodeOutcome(BaseModel):
	node_name: str
	execution_time_seconds: float
	input: str
	outcome: str
	args: dict[str, Any]


class ResearcherNodeOutcome(BaseNodeOutcome):
    intermediate_steps: List[BaseNodeOutcome] = Field(description="List of intermediate steps.")
    node_name: str = "agent_2_researcher"
    


class PipelineOutcome(BaseModel):
	query_id: int
	success: bool
	error: str | None
	nodes: dict[str, Union[BaseNodeOutcome, ResearcherNodeOutcome]] = Field(description="Dictionary of node outcomes.")
