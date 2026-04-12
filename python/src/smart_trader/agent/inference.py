"""Inference service — serves trained agent for live trading.

Supports PyTorch, ONNX Runtime, and TorchScript backends.
Includes shadow mode for paper validation of agent decisions
alongside a running rule-based baseline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class InferenceBackend(StrEnum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"


@dataclass
class InferenceConfig:
    backend: InferenceBackend = InferenceBackend.PYTORCH
    model_path: str = ""
    device: str = "cpu"
    max_latency_ms: float = 50.0
    shadow_mode: bool = False


@dataclass
class InferenceResult:
    action: dict
    value_estimate: float
    latency_ms: float
    backend: str


class InferenceService:
    """Serves agent inference for production trading."""

    def __init__(self, config: InferenceConfig) -> None:
        self.cfg = config
        self._model = None
        self._ort_session = None
        self._log = log.bind(backend=config.backend)

    async def start(self) -> None:
        """Load the model based on configured backend."""
        if self.cfg.backend == InferenceBackend.PYTORCH:
            self._load_pytorch()
        elif self.cfg.backend == InferenceBackend.ONNX:
            self._load_onnx()
        elif self.cfg.backend == InferenceBackend.TORCHSCRIPT:
            self._load_torchscript()
        self._log.info("inference_ready", model=self.cfg.model_path)

    def predict(self, observation: np.ndarray) -> InferenceResult:
        """Run inference on a single observation vector.

        Measures latency and logs warnings if exceeding threshold.
        """
        t0 = time.perf_counter()

        if self.cfg.backend == InferenceBackend.ONNX and self._ort_session:
            action, value = self._predict_onnx(observation)
        elif self.cfg.backend == InferenceBackend.TORCHSCRIPT and self._model:
            action, value = self._predict_torchscript(observation)
        elif self._model:
            action, value = self._predict_pytorch(observation)
        else:
            action = self._default_action()
            value = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000
        if latency_ms > self.cfg.max_latency_ms:
            self._log.warning("high_latency", latency_ms=f"{latency_ms:.1f}")

        return InferenceResult(
            action=action, value_estimate=value,
            latency_ms=latency_ms, backend=self.cfg.backend,
        )

    # ── backend loaders ────────────────────────────────────────

    def _load_pytorch(self) -> None:
        import torch
        from smart_trader.agent.meta_controller import MetaController

        ckpt = torch.load(self.cfg.model_path, map_location=self.cfg.device, weights_only=False)
        cfg = ckpt["config"]
        self._model = MetaController(
            obs_dim=cfg["obs_dim"],
            d_model=cfg["d_model"],
            n_layers=cfg.get("n_layers", 2),
            n_heads=cfg.get("n_heads", 4),
            device=self.cfg.device,
            lookback=cfg.get("lookback", 1),
            context_dim=cfg.get("context_dim", 0),
        )
        self._model.load(self.cfg.model_path)
        self._model.set_deterministic(True)

    def _load_onnx(self) -> None:
        import onnxruntime as ort
        self._ort_session = ort.InferenceSession(
            self.cfg.model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def _load_torchscript(self) -> None:
        import torch
        self._model = torch.jit.load(self.cfg.model_path, map_location=self.cfg.device)
        self._model.eval()

    # ── prediction methods ─────────────────────────────────────

    def _predict_pytorch(self, obs: np.ndarray) -> tuple[dict, float]:
        action, _, value = self._model.act(obs)
        return action, value

    def _predict_onnx(self, obs: np.ndarray) -> tuple[dict, float]:
        inputs = {"observation": obs.reshape(1, -1).astype(np.float32)}
        outputs = self._ort_session.run(None, inputs)
        action = {
            "position": np.clip(outputs[0].flatten(), -1, 1),
            "risk_budget": np.clip(outputs[1].flatten(), 0.01, 0.10),
            "hold_bars": int(outputs[2].argmax()),
        }
        value = float(outputs[4].item()) if len(outputs) > 4 else 0.0
        return action, value

    def _predict_torchscript(self, obs: np.ndarray) -> tuple[dict, float]:
        import torch
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            outputs = self._model(obs_t)
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            action = outputs[0]
            value = float(outputs[2].item()) if len(outputs) > 2 else 0.0
        else:
            action = self._default_action()
            value = 0.0
        return action, value

    @staticmethod
    def _default_action() -> dict:
        return {
            "position": np.array([0.0]),
            "risk_budget": np.array([0.02]),
            "hold_bars": 0,
        }


@dataclass
class ShadowRecord:
    timestamp: float
    agent_action: dict
    agent_value: float
    baseline_action: dict | None = None
    actual_outcome: float | None = None


class ShadowMode:
    """Runs the RL agent in parallel with a baseline, logging but not executing.

    Accumulates comparison records for offline analysis to validate
    the agent before going live.
    """

    def __init__(self, agent_service: InferenceService) -> None:
        self._agent = agent_service
        self._records: list[ShadowRecord] = []
        self._log = log.bind(mode="shadow")

    def evaluate(
        self,
        observation: np.ndarray,
        baseline_action: dict | None = None,
    ) -> ShadowRecord:
        """Run agent inference in shadow mode (no execution)."""
        result = self._agent.predict(observation)
        record = ShadowRecord(
            timestamp=time.time(),
            agent_action=result.action,
            agent_value=result.value_estimate,
            baseline_action=baseline_action,
        )
        self._records.append(record)
        self._log.debug(
            "shadow_decision",
            position=float(np.atleast_1d(result.action.get("position", 0))[0]),
            value=f"{result.value_estimate:.4f}",
            latency_ms=f"{result.latency_ms:.1f}",
        )
        return record

    def record_outcome(self, pnl: float) -> None:
        """Attach the actual market outcome to the last shadow record."""
        if self._records:
            self._records[-1].actual_outcome = pnl

    def get_summary(self) -> dict:
        """Compare agent vs baseline performance over shadow period."""
        if not self._records:
            return {"n_records": 0}

        outcomes = [r.actual_outcome for r in self._records if r.actual_outcome is not None]
        return {
            "n_records": len(self._records),
            "n_with_outcomes": len(outcomes),
            "agent_mean_value": float(np.mean([r.agent_value for r in self._records])),
        }

    def clear(self) -> None:
        self._records.clear()
