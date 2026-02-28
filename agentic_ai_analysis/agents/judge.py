import xml.etree.ElementTree as ET
from agentic_ai_analysis.core.llm_client import get_llm
from langchain_core.prompts import PromptTemplate
from typing import Dict, Any, Tuple

def parse_xml(xml_text: str) -> Dict[str, Any]:
    """Helper to parse the XML output from the Judge."""
    # Sometimes LLMs wrap the XML in markdown code blocks
    xml_text = xml_text.strip()
    if xml_text.startswith("```xml"):
        xml_text = xml_text[6:]
    if xml_text.startswith("```"):
        xml_text = xml_text[3:]
    if xml_text.endswith("```"):
        xml_text = xml_text[:-3]
    xml_text = xml_text.strip()
    
    try:
        # Wrap in a root element just in case the LLM didn't
        if not xml_text.startswith("<root>"):
            xml_text = f"<root>\n{xml_text}\n</root>"
            
        root = ET.fromstring(xml_text)
        
        is_related_elem = root.find("is_related")
        explanation_elem = root.find("explanation")
        
        is_related_str = is_related_elem.text.strip().lower() if is_related_elem is not None else "false"
        is_related = is_related_str == "true" or is_related_str == "1"
        
        explanation = explanation_elem.text.strip() if explanation_elem is not None else ""
        
        return {
            "is_related": is_related,
            "explanation": explanation
        }
    except Exception as e:
        # Fallback on parsing error
        return {
            "is_related": False,
            "explanation": f"XML parsing failed: {str(e)}\nRaw Response: {xml_text}"
        }

def run_judge(original_prompt: str, context: str) -> Tuple[str, Dict[str, Any]]:
    """
    Agent 4: Judge / Context Filter.
    Evaluates whether the provided context is relevant to the original prompt.
    Returns the raw XML response (for embedding) and the parsed results.
    """
    llm = get_llm()
    
    template = """You are a Judge Agent evaluating context retrieved for a query.
Evaluate whether the following context is highly related and useful for answering the prompt.

Original Prompt: {prompt}

Retrieved Context:
{context}

Respond ONLY in the following XML format. Do not add any text before or after the XML.
<reasoning>
Brief step-by-step thinking about relevance.
</reasoning>
<is_related>TRUE OR FALSE</is_related>
<explanation>Detailed explanation of why it is related and what specific useful facts it contains, or why it is unrelated.</explanation>
"""

    prompt_template = PromptTemplate(input_variables=["prompt", "context"], template=template)
    chain = prompt_template | llm
    
    response = chain.invoke({
        "prompt": original_prompt,
        "context": context
    })
    
    xml_text = str(response.content).strip()
    parsed_xml = parse_xml(xml_text)
    
    return xml_text, parsed_xml
