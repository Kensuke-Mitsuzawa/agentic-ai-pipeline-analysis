from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import yaml
from pydantic import BaseModel, Field


class LangfuseSettings(BaseModel):
    enabled: bool = Field(default=False, description="Enable Langfuse tracing.")
    host: str = Field(default="", description="Langfuse host URL (on-prem).")
    public_key: str = Field(default="", description="Langfuse public key.")
    secret_key: str = Field(default="", description="Langfuse secret key.")

    @staticmethod
    def load(config_path: Optional[Path] = None) -> "LangfuseSettings":
        # 1) env vars (preferred)
        host = os.environ.get("LANGFUSE_HOST", "")
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
        enabled_env = os.environ.get("LANGFUSE_ENABLED")

        # 2) optional config file
        if config_path is None:
            config_path = Path(os.environ.get("LLM_OPS_CONFIG_PATH", "llm_ops.yaml"))
        file_cfg: Dict[str, Any] = {}
        if config_path.exists():
            raw = yaml.safe_load(config_path.read_text()) or {}
            if isinstance(raw, dict):
                file_cfg = raw.get("langfuse", raw)

        enabled = False
        if enabled_env is not None:
            enabled = enabled_env.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(file_cfg.get("enabled"), bool):
            enabled = bool(file_cfg["enabled"])
        else:
            # auto-enable if keys present
            enabled = bool(host and public_key and secret_key)

        return LangfuseSettings(
            enabled=enabled,
            host=str(file_cfg.get("host") or host or ""),
            public_key=str(file_cfg.get("public_key") or public_key or ""),
            secret_key=str(file_cfg.get("secret_key") or secret_key or ""),
        )


@dataclass
class TraceHandle:
    trace_id: str
    _client: Any
    _trace: Any


class _NoopTracer:
    def start_trace(self, *, trace_id: str, name: str, input: Any, metadata: Optional[dict] = None) -> TraceHandle:
        return TraceHandle(trace_id=trace_id, _client=None, _trace=None)

    @contextmanager
    def span(self, trace: TraceHandle, *, name: str, input: Any, metadata: Optional[dict] = None) -> Iterator[None]:
        yield

    def score(self, trace: TraceHandle, *, name: str, value: float, comment: Optional[str] = None) -> None:
        return


class _LangfuseTracer:
    def __init__(self, settings: LangfuseSettings):
        from langfuse import Langfuse  # imported lazily

        self._client = Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            host=settings.host,
        )

    def start_trace(self, *, trace_id: str, name: str, input: Any, metadata: Optional[dict] = None) -> TraceHandle:
        trace = self._client.trace(id=trace_id, name=name, input=input, metadata=metadata or {})
        return TraceHandle(trace_id=trace_id, _client=self._client, _trace=trace)

    @contextmanager
    def span(self, trace: TraceHandle, *, name: str, input: Any, metadata: Optional[dict] = None) -> Iterator[None]:
        if trace._trace is None:
            yield
            return
        span = trace._trace.span(name=name, input=input, metadata=metadata or {})
        try:
            yield
        except Exception as e:
            span.update(status="ERROR", metadata={"error": str(e)})
            raise
        finally:
            span.end()

    def score(self, trace: TraceHandle, *, name: str, value: float, comment: Optional[str] = None) -> None:
        if trace._trace is None:
            return
        trace._trace.score(name=name, value=value, comment=comment)


_TRACER_SINGLETON: Any = None


def get_tracer() -> Any:
    global _TRACER_SINGLETON
    # If env/config changes between tests, allow a reset by setting this env var.
    if os.environ.get("LANGFUSE_TRACER_RESET") == "1":
        _TRACER_SINGLETON = None
    if _TRACER_SINGLETON is not None:
        return _TRACER_SINGLETON

    settings = LangfuseSettings.load()
    if not settings.enabled:
        _TRACER_SINGLETON = _NoopTracer()
        return _TRACER_SINGLETON

    try:
        _TRACER_SINGLETON = _LangfuseTracer(settings)
    except Exception:
        # Never break the pipeline if tracing misconfigured.
        _TRACER_SINGLETON = _NoopTracer()
    return _TRACER_SINGLETON

