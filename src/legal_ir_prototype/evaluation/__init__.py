"""Reproducible offline evaluation for the online legal RAG pipeline."""

from .dataset import load_cases, prepare_dataset
from .deterministic import run_deterministic_checks
from .recorder import ArtifactRecorder
from .schemas import (
    DatasetManifest,
    DeterministicEvaluation,
    EvalCase,
    EvaluationError,
    EvaluationPrediction,
    JudgeAssessment,
    JudgeResult,
)

__all__ = [
    "ArtifactRecorder",
    "DatasetManifest",
    "DeterministicEvaluation",
    "EvalCase",
    "EvaluationError",
    "EvaluationPrediction",
    "JudgeAssessment",
    "JudgeResult",
    "load_cases",
    "prepare_dataset",
    "run_deterministic_checks",
]
