from langchain_openai import ChatOpenAI
import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from pydantic import BaseModel, Field

# Using HuggingFaceEndpoint for a local vLLM or TGI server.
# By default, assuming Mistral-7B-Instruct served on a local port (e.g., 8000).
# The embedding model is loaded locally via HuggingFaceEmbeddings.

class LlmGenerationParameters(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 999
    top_p: float = 0.9
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] = []


def get_llm(
    base_url="http://localhost:8000/v1/", 
    model="local-model", 
    generation_parameters: LlmGenerationParameters = LlmGenerationParameters()
) -> ChatOpenAI:
    """
    Returns a LangChain LLM connected to a local vLLM/FastAPI OpenAI-compatible endpoint.
    If testing without a server, users can use the actual HuggingFace Hub inference API
    by providing a standard HF endpoint URL and an API key.
    """
    # For a real local server we use a generic placeholder like HuggingFaceEndpoint
    # Alternatively, if serving via OpenAI compatible endpoints (vLLM):
    from langchain_openai import ChatOpenAI
    
    # Allow environment override so interface scripts can switch endpoints/models
    # without editing code (e.g., when using port-forward to a remote machine).
    base_url = os.environ.get("OPENAI_API_BASE", base_url)
    model = os.environ.get("OPENAI_MODEL", model)

    # We use ChatOpenAI pointing to the local vLLM server since it exposes standard API
    llm = ChatOpenAI(
        model=model,
        temperature=generation_parameters.temperature,
        max_tokens=generation_parameters.max_tokens,
        top_p=generation_parameters.top_p,
        presence_penalty=generation_parameters.presence_penalty,
        frequency_penalty=generation_parameters.frequency_penalty,
        stop=generation_parameters.stop,
        openai_api_key="EMPTY",  # Local endpoint doesn't need key
        openai_api_base=base_url,
        timeout=60, # Add timeout to prevent freezing
    )
    return llm

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a LangChain Embeddings interface using a local sentence-transformers model.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
