import logging
import time
import json
import ast
from typing import Dict, Any, List, TypedDict

from langchain_core.prompts import PromptTemplate
from agentic_ai_analysis.core.llm_client import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from .data_models import ResearcherNodeOutcome, BaseNodeOutcome, PromptContext


logger = logging.getLogger(__name__)

PossibleNodeNmaes = [
    "agent_2_node_a_retriever",
    "agent_2_node_b_filter",
    "agent_2_node_c_judge",
    "agent_2_researcher",
]


def parse_structured_query(query: str):
    """
    Tries to parse the query as a PromptContext JSON or a tuple/list.
    Returns (arxiv_id, options, question) or (None, [], query).
    """
    try:
        # 1. Try to parse as PromptContext JSON (New preferred way)
        context = PromptContext.model_validate_json(query)
        return context.arxiv_id, context.options or [], context.question
    except Exception:
        pass

    try:
        # 2. Try as python literal (handles tuples/lists in string form)
        parsed = ast.literal_eval(query)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
            return str(parsed[0]), parsed[1], str(parsed[2])
    except:
        pass
        
    try:
        # 3. Try to parse as raw JSON list
        data = json.loads(query)
        if isinstance(data, list) and len(data) >= 3:
            return str(data[0]), data[1], str(data[2])
    except:
        pass
        
    return None, [], query


def fetch_arxiv_abstract(arxiv_id: str) -> str:
    """
    Fetches the abstract of a paper from Arxiv using its ID.
    """
    arxiv = ArxivAPIWrapper()
    try:
        # ArxivAPIWrapper.run with an ID returns the paper summary
        return arxiv.run(arxiv_id)
    except Exception as e:
        logger.error(f"Failed to fetch Arxiv abstract for {arxiv_id}: {e}")
        return ""


class ResearcherState(TypedDict):
    user_query: str
    search_keywords: List[str]
    raw_documents: List[str]
    filtered_tuples: List[Dict[str, str]]
    iteration_count: int


def run_researcher(
    user_query: str,
    node_order: int = 0, 
    max_depth: int = 3) -> ResearcherNodeOutcome:
    """
    Agent 2: Modular State-Machine Researcher.
    Uses Nodes A (Retriever), B (Filter), C (Judge) to iteratively
    retrieve Arxiv docs, filter signals, and decide when to stop.
    """
    node_order_inner = 0

    start_time = time.perf_counter()
    llm = get_llm()
    arxiv_tool = ArxivQueryRun()

    # Parse structured input
    arxiv_id, options, question = parse_structured_query(user_query)
    abstract = ""
    if arxiv_id:
        abstract = fetch_arxiv_abstract(arxiv_id)
        if abstract:
            logger.info(f"Successfully fetched abstract for paper {arxiv_id}")
        else:
            logger.warning(f"Could not fetch abstract for paper {arxiv_id}")

    # Initial Extraction
    if abstract:
        init_extract_prompt = PromptTemplate.from_template(
            "You are a helpful academic keyword extractor.\n"
            "Below is the abstract of an Arxiv paper and a question about it.\n\n"
            "Abstract: {abstract}\n\n"
            "Question: {question}\n"
            "Options: {options}\n\n"
            "Extract the most important keywords to search for additional research papers that would help answer the question accurately.\n"
            "Return ONLY a comma-separated list of keywords.\n"
            "Do not provide any conversational text or explanation.\n\n"
            "Keywords:"
        )
        prompt_vars = {
            "abstract": abstract,
            "question": question,
            "options": str(options)
        }
    else:
        init_extract_prompt = PromptTemplate.from_template(
            "You are a helpful academic keyword extractor. "
            "Given the user prompt, extract the most important keywords and return ONLY a comma-separated list.\n"
            "Do not provide any conversational text or explanation.\n\n"
            "Prompt: {prompt}\nKeywords:"
        )
        prompt_vars = {"prompt": user_query}

    init_chain = init_extract_prompt | llm
    init_kws_str = str(init_chain.invoke(prompt_vars).content).strip()
    logger.info(f"Initial keywords: {init_kws_str}")
    
    # We enrich the internal user_query with options for better filtering/judging
    effective_query = f"{question} Options: {options}" if options else (question if arxiv_id else user_query)

    state: ResearcherState = {
        "user_query": effective_query,
        "search_keywords": [k.strip() for k in init_kws_str.split(",") if k.strip()],
        "raw_documents": [],
        "filtered_tuples": [],
        "iteration_count": 0
    }
    
    # Prompts for Nodes
    query_prompt = PromptTemplate.from_template(
        "Based on these keywords: {keywords}, formulate a single concise search query for Arxiv to find the most relevant papers. Only output the query string."
    )
    
    filter_prompt = PromptTemplate.from_template(
        "Read the following raw documents retrieved from Arxiv:\n{docs}\n\n"
        "User Query: {user_query}\n\n"
        "Extract only information highly relevant to the user query. "
        "Return the output in strictly formatted XML. Provide a list of <item> elements inside a root <results> element. "
        "Each <item> must contain a <keyword> and a <description> element.\n"
        "Example:\n"
        "<results>\n"
        "  <item>\n"
        "    <keyword>...</keyword>\n"
        "    <description>...</description>\n"
        "  </item>\n"
        "</results>\n"
        "Output ONLY valid XML without any text outside the XML block."
    )
    
    judge_prompt = PromptTemplate.from_template(
        "User Query: {user_query}\n\n"
        "Current Synthesized Data:\n{filtered_data}\n\n"
        "Evaluate if the current synthesized data contains sufficient detail to definitively answer the user's query.\n"
        "First provide your reasoning in a 'Reasoning:' section.\n"
        "Then, on a new line, provide a strict categorical label: 'Label: SUFFICIENT' or 'Label: INSUFFICIENT'."
    )
    
    next_kws_prompt = PromptTemplate.from_template(
        "The following information is insufficient to fully answer the query.\n"
        "User Query: {user_query}\n"
        "Current Data: {filtered_data}\n"
        "Reasoning for insufficiency: {judge_reasoning}\n\n"
        "Generate a comma-separated list of new keyword terms to search next to fill the gaps. "
        "Return ONLY a comma-separated list."
    )

    paired_steps = []
    
    while state["iteration_count"] < max_depth:
        # Node A: Retriever
        current_kws_str = ", ".join(state["search_keywords"])
        query_chain = query_prompt | llm
        arxiv_query = str(query_chain.invoke({"keywords": current_kws_str}).content).strip()
        
        try:
            retrieved_docs = arxiv_tool.invoke({"query": arxiv_query})
            logger.debug(f"Retrieved docs: {retrieved_docs}")
        except Exception as e:
            logger.info(f"Error retrieving docs: {str(e)}")
            retrieved_docs = f"Error retrieving docs: {str(e)}"
        # end try
        
        state["raw_documents"].append(str(retrieved_docs))
        
        paired_steps.append(BaseNodeOutcome(
            node_order=node_order_inner,
            node_name="agent_2_node_a_retriever",
            execution_time_seconds=0,
            input=current_kws_str,
            outcome=f"Query: {arxiv_query}\nDocs: {retrieved_docs}",
            args={}
        ))
        node_order_inner += 1
        
        # Node B: Filter
        filter_chain = filter_prompt | llm
        filter_res = str(filter_chain.invoke({
            "docs": state["raw_documents"][-1], 
            "user_query": state["user_query"]
        }).content).strip()
        
        # Parse XML
        try:
            # Clean up potential markdown formatting from LLM XML output
            clean_xml = filter_res
            if "```xml" in clean_xml:
                clean_xml = clean_xml.split("```xml", 1)[1]
            if "```" in clean_xml:
                clean_xml = clean_xml.rsplit("```", 1)[0]
            clean_xml = clean_xml.strip()
            
            import xml.etree.ElementTree as ET
            # Sometimes models return extra text, try to find the <results> block
            start_idx = clean_xml.find("<results>")
            end_idx = clean_xml.rfind("</results>")
            if start_idx != -1 and end_idx != -1:
                clean_xml = clean_xml[start_idx:end_idx + 10]
            
            root = ET.fromstring(clean_xml)
            for item in root.findall('.//item'):
                kw_node = item.find('keyword')
                desc_node = item.find('description')
                if kw_node is not None and desc_node is not None and kw_node.text and desc_node.text:
                    state["filtered_tuples"].append({
                        "keyword": kw_node.text.strip(),
                        "description": desc_node.text.strip()
                    })
        except Exception as e:
            logger.error(f"Error filtering docs: {str(e)}")
            logger.error(f"Filter result: {filter_res}")
            pass # Or handle gracefully
        # end try

        logger.debug(f"Filtered tuples: {state['filtered_tuples']}")
        # Clear raw documents after filtering to optimize context
        state["raw_documents"].clear()
        
        paired_steps.append(BaseNodeOutcome(
            node_order=node_order_inner,
            node_name="agent_2_node_b_filter",
            execution_time_seconds=0,
            input=retrieved_docs,
            outcome=filter_res, # The raw text is saved for CKA
            args={}
        ))
        node_order_inner += 1
        
        # Node C: Judge
        filtered_data_str = json.dumps(state["filtered_tuples"], indent=2)
        judge_chain = judge_prompt | llm
        judge_res = str(judge_chain.invoke({
            "user_query": state["user_query"],
            "filtered_data": filtered_data_str
        }).content).strip()
        logger.debug(f"Judge result: {judge_res}")
        
        paired_steps.append(BaseNodeOutcome(
            node_order=node_order_inner,
            node_name="agent_2_node_c_judge",
            execution_time_seconds=0,
            input=filtered_data_str,
            outcome=judge_res, # Textual reasoning + label for CKA
            args={}
        ))
        node_order_inner += 1
        
        # Control Flow & Routing Logic
        judge_lines = judge_res.split("\n")
        label = "INSUFFICIENT"
        for line in judge_lines:
            if "Label:" in line:
                if "SUFFICIENT" in line and "INSUFFICIENT" not in line:
                    label = "SUFFICIENT"
                break
            # end if
        # end for
        if label == "SUFFICIENT":
            break
        # end if
        

        # INSUFFICIENT case
        state["iteration_count"] += 1
        
        if state["iteration_count"] < max_depth:
            # Generate new search keywords
            next_kws_chain = next_kws_prompt | llm
            next_kws_str = str(next_kws_chain.invoke({
                "user_query": state["user_query"],
                "filtered_data": filtered_data_str,
                "judge_reasoning": judge_res
            }).content).strip()
            state["search_keywords"] = [k.strip() for k in next_kws_str.split(",") if k.strip()]
        # end if
    # end while
    
    final_execution_time = time.perf_counter() - start_time
    # The output passed to synthesizer is the accumulated filtered_tuples
    final_answer = json.dumps(state["filtered_tuples"])
    
    logger.debug(f"Final answer: {final_answer}")
    return ResearcherNodeOutcome(
        node_order=node_order,
        node_name="agent_2_researcher",
        execution_time_seconds=final_execution_time,
        input=user_query,
        outcome=final_answer,
        args={},
        intermediate_steps=paired_steps
    )
