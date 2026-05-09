# Agentic AI Pipeline Analysis

A research-oriented framework for building, evaluating, and observing multi-agent AI pipelines. Designed for deep analysis of agentic workflows, including internal representation tracking and distribution shift detection.

## What it is
This repository provides a robust environment to prototype and analyze Agentic AI systems. It leverages **LangGraph** for orchestration and **Langfuse** for full-stack observability. The framework is specifically tuned for scientific question-answering tasks (e.g., SciQ, ARC) where multiple specialized agents cooperate to retrieve, synthesize, and judge information.

## What is implemented
- **Multi-Agent Orchestration**: A modular state machine architecture using LangGraph.
- **Automated Research**: Agents capable of fetching and filtering context from academic sources (arXiv).
- **Outcome Evaluation**: Integrated "Judge" agents that score pipeline results for correctness and relevance.
- **Local LLM Serving**: A built-in FastAPI server that wraps HuggingFace models with `bitsandbytes` 4-bit quantization support.
- **Deep Observability**: Native Langfuse integration for tracing every step of the agentic process, including metadata and evaluation scores.
- **Advanced Metrics**: Support for Centered Kernel Alignment (CKA) to analyze internal model representations across the pipeline.

## Agents & Components
The pipeline is composed of specialized "middlewares" that handle distinct parts of the reasoning chain:

- **`ResearcherAgent`**: Responsible for information retrieval. It queries external APIs (like arXiv) and filters results based on query relevance.
- **`SynthesizerAgent`**: The generation core. It takes retrieved context and synthesizes a final response, ensuring grounding in the provided facts.
- **`JudgeAgent`**: An evaluation specialist. It compares the model output against ground truth or heuristics to provide structured feedback and scores.
- **`Orchestrator`**: The backbone that manages state transitions between agents, handling retries and conditional branching.
- **`LangfuseTracer`**: An observability middleware that captures prompts, completions, latencies, and costs.

## Installation

This project uses `uv` for lightning-fast dependency management.

### Prerequisites
- **Python**: 3.12+ (managed by `uv`)
- **GPU**: Recommended for local LLM serving (CUDA 12+).

### Setup
1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and Sync**:
   ```bash
   git clone <repository-url>
   cd agentic-ai-pipeline-analysis
   uv sync
   ```

3. **Environment Variables**:
   Copy `.env_base` to `.env` and fill in your configuration:
   ```bash
   cp .env_base .env
   ```

## Guide to Deploy the LLM Server

The project includes a built-in OpenAI-compatible server to run open-source models locally.

### 1. Automatic Deployment (Managed)
The `interface_job.py` entrypoint can automatically launch and manage the LLM server lifecycle based on your config. In your `.toml` config file, ensure:
```toml
[local_server]
start = true
model_id = "Qwen/Qwen2.5-7B-Instruct"
port = 8000
host = "127.0.0.1"
```

### 2. Standalone Deployment
If you wish to run the server independently:
```bash
uv run python -m agentic_ai_analysis.core.local_server
```
The server supports automatic 4-bit quantization to fit larger models on consumer GPUs.

## Usage

Run the evaluation pipeline against a dataset:

```bash
uv run python interface/interface_job.py -c interface/app_configs/config_local.toml -e .env -n 10
```

- `-c`: Path to your TOML configuration.
- `-e`: Path to your `.env` file.
- `-n`: Number of queries to process from the dataset.

### Observability
To view traces, ensure your Langfuse instance is running (see `langfuse-onprem/`) and configured in your `.env`.
