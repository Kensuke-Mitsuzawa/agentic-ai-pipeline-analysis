from pydantic import BaseModel, Field
from typing import Dict, Any, List, Union, Optional


class PromptContext(BaseModel):
	context_id: Optional[str] = None
	options: Optional[List[str]] = None
	question: str


class BaseNodeOutcome(BaseModel):
	node_order: int
	node_name: str
	execution_time_seconds: float
	input: str
	outcome: str
	args: dict[str, Any]


class ResearcherNodeOutcome(BaseNodeOutcome):
    intermediate_steps: List[BaseNodeOutcome] = Field(description="List of intermediate steps.")
    node_name: str = "agent_2_researcher"
    


class PipelineOutcome(BaseModel):
	query_id: str
	prompt: str
	final_outcome: str
	success: bool
	error: str | None
	nodes: dict[str, Union[BaseNodeOutcome, ResearcherNodeOutcome]] = Field(description="Dictionary of node outcomes.")

	def get_node_names(self) -> List[str]:
		_set_node_obj = list(sorted(self.nodes.values(), key=lambda x: x.node_order))
		return [n.node_name for n in _set_node_obj]
