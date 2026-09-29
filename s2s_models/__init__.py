"""S2S prediction baselines, feature extraction, and evaluation metrics."""

from s2s_models.baselines import (
    ClimatologyBaseline,
    FixedCorrectionBaseline,
    RawEnsembleBaseline,
    SpreadGatedBaseline,
)
from s2s_models.features import DatasetLoader, extract_features_for_lead_week
from s2s_models.metrics import (
    brier_score,
    brier_skill_score,
    compute_all_metrics,
    reliability_curve_data,
)
from s2s_models.neural_gated import JointConfidenceGatedNet, NeuralGatedBaseline

__all__ = [
    "ClimatologyBaseline",
    "DatasetLoader",
    "FixedCorrectionBaseline",
    "JointConfidenceGatedNet",
    "NeuralGatedBaseline",
    "RawEnsembleBaseline",
    "SpreadGatedBaseline",
    "brier_score",
    "brier_skill_score",
    "compute_all_metrics",
    "extract_features_for_lead_week",
    "reliability_curve_data",
]


