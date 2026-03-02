import os
import time
import json
import logging
import multiprocessing
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LocalServerConfig(BaseModel):
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2"
    port: int = 8000
    host: str = "127.0.0.1"
    # We store the config as a dictionary to avoid Pydantic validation issues with the transformers object
    # In practice this should be a dictionary representing the kwargs for BitsAndBytesConfig
    quantization_config_dict: Optional[Dict[str, Any]] = None

# Global pipeline instance for the worker process
_pipeline = None

def init_pipeline(config_json: str):
    """
    Initializes the HuggingFace pipeline in the worker process.
    """
    global _pipeline
    config = LocalServerConfig.model_validate_json(config_json)
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    model_kwargs = {"device_map": "auto"}
    
    if config.quantization_config_dict:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(**config.quantization_config_dict)
        model_kwargs["quantization_config"] = bnb_config  # type: ignore
        
    logger.info(f"Loading model {config.model_id} with kwargs: {model_kwargs}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    
    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        return_full_text=False
    )
    logger.info("Model loaded successfully.")

app = FastAPI()

@app.get("/health")
def health_check():
    if _pipeline is not None:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Model loading")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    data = await request.json()
    messages = data.get("messages", [])
    
    if not messages:
        raise HTTPException(status_code=400, detail="Missing messages")
        
    # very basic chat template string formatting since older models don't all support apply_chat_template perfectly
    # For mistral, usually tokenizer.apply_chat_template works. We'll try it.
    try:
        prompt = _pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback dump
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages]) + "\nassistant: "
        
    # Generate
    outputs = _pipeline(prompt, max_new_tokens=data.get("max_tokens", 512), temperature=data.get("temperature", 0.7))
    generated_text = outputs[0]["generated_text"].strip()
    
    # Return OpenAI compatible schema
    response = {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": data.get("model", "local-model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
    }
    return response

def _run_server(
    config_json: str, 
    error_queue: Optional[multiprocessing.Queue] = None, 
    env_dict: Optional[Dict[str, str]] = None
) -> None:
    """Entry point for the multiprocessing.Process"""
    if env_dict:
        os.environ.update(env_dict)

    print(f"DEBUG child process HF_HOME: {os.environ.get('HF_HOME')}")
    try:
        # Initialize the model before starting the server so health-check is truthful
        config = LocalServerConfig.model_validate_json(config_json)
        init_pipeline(config_json)
        # Return the status of the launching server.
        uvicorn.run(app, host=config.host, port=config.port, log_level="warning")
    except Exception as e:
        logger.error(f"Local server failed to start: {e}")
        if error_queue is not None:
            error_queue.put(str(e))


class LocalServerManager:
    def __init__(self, config: LocalServerConfig):
        self.config = config
        self.process = None

    def start(self):
        """Starts the server in a new process and waits for it to be healthy."""
        logger.info(f"Starting local server on {self.config.host}:{self.config.port}...")
        
        self.error_queue = multiprocessing.Queue()
        
        # Start the FastAPI runner process
        
        # We need to explicitly pass the loaded environment variables to the 
        # multiprocessing worker so they aren't lost to default sub-shell env setup.
        env_dict = dict(os.environ)
        self.process = multiprocessing.Process(
            target=_run_server,
            args=(self.config.model_dump_json(), self.error_queue, env_dict),
            daemon=False
        )
        self.process.start()
        
        # Poll for health
        import requests
        import time
        import queue
        
        url = f"http://{self.config.host}:{self.config.port}/health"
        max_retries = 60
        for i in range(max_retries):
            if not self.process.is_alive():
                logger.error("Local server process terminated unexpectedly.")
                
                try:
                    error_msg = self.error_queue.get_nowait()
                except queue.Empty:
                    error_msg = "Process exited without reporting an error."
                
                self.process = None
                raise RuntimeError(f"Local server failed to start: {error_msg}")

            try:
                resp = requests.get(url, timeout=2)
                if resp.status_code == 200:
                    logger.info("Local server is healthy and ready.")
                    return
            except requests.exceptions.RequestException:
                pass
                
            time.sleep(5)
            logger.info(f"Waiting for server... ({i+1}/{max_retries})")
            
        self.stop()
        raise RuntimeError("Local server failed to start or load model within timeout.")

    def stop(self):
        """Terminates the server process."""
        if self.process and self.process.is_alive():
            logger.info("Stopping local server...")
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
            logger.info("Local server stopped.")
            self.process = None

# Global easy-access manager
_manager = None

def start_local_server(config: LocalServerConfig):
    global _manager
    if _manager is not None:
        logger.warning("Local server is already running. Stopping it first.")
        stop_local_server()
        
    _manager = LocalServerManager(config)
    _manager.start()

def stop_local_server():
    global _manager
    if _manager is not None:
        _manager.stop()
        _manager = None
