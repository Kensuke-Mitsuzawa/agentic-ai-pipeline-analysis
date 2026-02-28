import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings

# Using HuggingFaceEndpoint for a local vLLM or TGI server.
# By default, assuming Mistral-7B-Instruct served on a local port (e.g., 8000).
# The embedding model is loaded locally via HuggingFaceEmbeddings.

def get_llm(base_url="http://localhost:8000/v1/", model="mistralai/Mistral-7B-Instruct-v0.2"):
    """
    Returns a LangChain LLM connected to a local vLLM/TGI OpenAI-compatible endpoint.
    If testing without a server, users can use the actual HuggingFace Hub inference API
    by providing a standard HF endpoint URL and an API key.
    """
    # For a real local server we use a generic placeholder like HuggingFaceEndpoint
    # Alternatively, if serving via OpenAI compatible endpoints (vLLM):
    from langchain_openai import ChatOpenAI
    
    # We use ChatOpenAI pointing to the local vLLM server since it exposes standard API
    llm = ChatOpenAI(
        model=model,
        temperature=0.7,
        max_tokens=512,
        openai_api_key="EMPTY",  # Local endpoint doesn't need key
        openai_api_base=base_url
    )
    return llm

def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Returns a LangChain Embeddings interface using a local sentence-transformers model.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
