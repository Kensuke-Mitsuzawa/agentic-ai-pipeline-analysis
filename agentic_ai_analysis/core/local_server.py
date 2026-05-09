import os
import time
import json
import logging
import multiprocessing
import threading
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

logger = logging.getLogger(__name__)


AcceptableLLMs = Literal[
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "Qwen/Qwen3.5-27B-FP8", 
    "Qwen/Qwen3.5-14B", 
    "default",
    "dummy"
]

class LLMClientConfig(BaseModel):
    openai_api_base: str = Field(..., description="The base URL of the OpenAI API.")
    model: AcceptableLLMs = "default"


class LocalServerConfig(BaseModel):
    model_id: AcceptableLLMs = "Qwen/Qwen2.5-7B-Instruct"
    port: int = 8000
    host: str = "127.0.0.1"
    start: bool = Field(default=False, description="If true, launch the local server process/thread.")
    # Default 4-bit quantization configuration for optimized local inference
    quantization_config_dict: Dict[str, Any] = Field(
        default_factory=lambda: {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4"
        }
    )

# Global pipeline instance for the worker process
_pipeline = None
_current_config: Optional[LocalServerConfig] = None
_default_model_id: Optional[str] = None
_reload_lock = threading.Lock()

class _DummyPipeline:
    def __init__(self):
        self.tokenizer = None

    def __call__(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0):
        # Deterministic tiny response for offline/unit tests.
        text = "DUMMY_RESPONSE"
        return [{"generated_text": text}]

def init_pipeline(config: LocalServerConfig):
    """
    Initializes the HuggingFace pipeline in the worker process.
    """
    global _pipeline, _current_config, _default_model_id
    
    # Cleanup previous model if exists
    if _pipeline is not None:
        logger.info("Cleaning up previous model...")
        _pipeline = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if config.model_id.startswith("dummy"):
        _pipeline = _DummyPipeline()
        _current_config = config
        if _default_model_id is None:
            _default_model_id = config.model_id
        logger.info(f"Initialized dummy pipeline for {config.model_id} (no HF downloads).")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    
    model_kwargs: Dict[str, Any] = {"device_map": "auto"}
    
    if config.quantization_config_dict:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(**config.quantization_config_dict)
        model_kwargs["quantization_config"] = bnb_config  # type: ignore
        
    logger.info(f"Loading model {config.model_id} with kwargs: {model_kwargs}")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    # Prefer safetensors to avoid torch.load restrictions on older torch versions.
    try:
        model = AutoModelForCausalLM.from_pretrained(config.model_id, use_safetensors=True, **model_kwargs)
    except Exception as e:
        # Fallback only if safetensors isn't available. If torch.load is blocked, surface a clear message.
        err = str(e)
        if "torch.load" in err and "upgrade torch" in err:
            raise RuntimeError(
                "Model load failed due to torch.load security restriction (torch<2.6) and non-safetensors weights. "
                "Use a model that provides safetensors (recommended), or upgrade torch to >=2.6."
            ) from e
        model = AutoModelForCausalLM.from_pretrained(config.model_id, use_safetensors=False, **model_kwargs)
    
    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        return_full_text=False
    )
    _current_config = config
    if _default_model_id is None:
        _default_model_id = config.model_id
    logger.info(f"Model {config.model_id} loaded successfully.")

app = FastAPI()

@app.get("/v1/models")
def list_models():
    # Minimal OpenAI-compatible models endpoint.
    model_id = "local-model"
    # In dummy mode, config isn't available here; return a generic entry.
    return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

@app.get("/health")
def health_check():
    if _pipeline is not None:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="Model loading")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """
    Minimal OpenAI-compatible embeddings endpoint.
    For now, returns deterministic zero vectors (sufficient for wiring + tests).
    """
    data = await request.json()
    inp = data.get("input")
    if inp is None:
        raise HTTPException(status_code=400, detail="Missing input")
    if isinstance(inp, str):
        inputs = [inp]
    else:
        inputs = list(inp)

    dim = int(data.get("dimensions") or 8)
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.0] * dim}
            for i in range(len(inputs))
        ],
        "model": data.get("model", "local-model"),
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible chat completions endpoint.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
        
    data = await request.json()
    requested_model = data.get("model", "default")

    # Check if we need to reload
    if _current_config is not None:
        target_model_id = _default_model_id if requested_model == "default" else requested_model
        
        if target_model_id != _current_config.model_id:
            logger.info(f"Request for model '{requested_model}' triggers reload. Current: '{_current_config.model_id}', Target: '{target_model_id}'")
            with _reload_lock:
                # Check again inside lock
                if target_model_id != _current_config.model_id:
                    new_config = _current_config.model_copy()
                    new_config.model_id = target_model_id
                    try:
                        init_pipeline(new_config)
                    except Exception as e:
                        logger.error(f"Failed to reload model {target_model_id}: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")
    messages = data.get("messages", [])
    
    if not messages:
        raise HTTPException(status_code=400, detail="Missing messages")
        
    # very basic chat template string formatting since older models don't all support apply_chat_template perfectly
    # For mistral, usually tokenizer.apply_chat_template works. We'll try it.
    try:
        if getattr(_pipeline, "tokenizer", None) is None:
            raise RuntimeError("no_tokenizer")
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
        init_pipeline(config)
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
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None

    def start(self):
        """Starts the server in a new process and waits for it to be healthy."""
        logger.info(f"Starting local server on {self.config.host}:{self.config.port}...")
        
        self.error_queue = multiprocessing.Queue()

        # Dummy/offline mode: run uvicorn in-process thread (no signals required to stop).
        if self.config.model_id == "dummy":
            try:
                init_pipeline(self.config)
            except Exception as e:
                raise RuntimeError(f"Dummy local server failed to init pipeline: {e}") from e

            uv_config = uvicorn.Config(app, host=self.config.host, port=self.config.port, log_level="warning")
            self._uvicorn_server = uvicorn.Server(uv_config)
            self._thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
            self._thread.start()
            # fall through to health polling below (process remains None)
        else:
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
            if self.process is not None and (not self.process.is_alive()):
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
        if self._uvicorn_server is not None:
            logger.info("Stopping local server (thread)...")
            self._uvicorn_server.should_exit = True
            if self._thread is not None:
                self._thread.join(timeout=10)
            self._uvicorn_server = None
            self._thread = None
            logger.info("Local server stopped.")
            return

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
