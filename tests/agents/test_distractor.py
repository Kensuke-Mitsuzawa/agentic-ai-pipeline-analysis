from agentic_ai_analysis.agents.distractor import run_distractor
from agentic_ai_analysis.core.local_server import LocalServerConfig, start_local_server, stop_local_server


def test_distractor():
    server_config = LocalServerConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct", # Super small model for fast test bootup
        port=8000,
        quantization_config_dict=dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16")
    )
    start_local_server(server_config)

    prompt = "Hello, how are you?"
    result = run_distractor(prompt)
    print(result)

    stop_local_server()

if __name__ == "__main__":
    test_distractor()