"""
heptapod.tools.dqm.models.classifier
=======================================
Tool for ML-based run quality classification.

While the autoencoder flags anomalous *histograms*, the classifier provides
a run-level quality label — GOOD, BAD, or UNCERTAIN — by aggregating
histogram-level signals into a single prediction. This mirrors the
certification workflow used by human shifters.

Architecture
------------
A lightweight gradient-boosted tree (scikit-learn GradientBoostingClassifier
or XGBoost) trained on hand-crafted features extracted by HistogramFeatureTool:
  * Mean / std of reconstruction errors (from autoencoder)
  * Number of flagged histograms (from reference comparison)
  * Fraction of dead / hot channels
  * Run duration and luminosity
  * Detector occupancy summary statistics

The model outputs calibrated probabilities for [BAD, UNCERTAIN, GOOD].

References
----------
* CMS ML group: https://cms-ml.github.io/documentation/
* Azzeh et al., "Machine learning for data quality monitoring", 2021
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class RunLabel(str, Enum):
    GOOD = "GOOD"
    UNCERTAIN = "UNCERTAIN"
    BAD = "BAD"


class ClassifierRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    feature_vector: list[float] = Field(
        ...,
        description=(
            "Flattened feature vector for a single run. "
            "Generate with HistogramFeatureTool."
        ),
    )
    model_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a saved scikit-learn / XGBoost model pickle. "
                    "None = use the pre-loaded model.",
    )
    uncertainty_band: float = Field(
        default=0.15,
        ge=0.0,
        le=0.5,
        description=(
            "Probability range around 0.5 classified as UNCERTAIN "
            "rather than GOOD or BAD."
        ),
    )


class ClassifierStateFields(BaseModel):
    """Agent state updated after each invocation."""

    last_prediction: Optional[str] = None
    last_probabilities: dict[str, float] = Field(default_factory=dict)


class RunQualityClassifierTool:
    """Classify a CMS run as GOOD / UNCERTAIN / BAD from a feature vector.

    Provides a concise run-level summary the agent can include in a
    DQMRunCard and communicate to shifters via ShifterAlertTool.

    Usage
    -----
    >>> tool = RunQualityClassifierTool()
    >>> result = tool.execute(
    ...     ClassifierRuntimeFields(feature_vector=[0.12, 3, 0.92, ...]),
    ...     ClassifierStateFields(),
    ... )
    """

    name: str = "run_quality_classifier"
    description: str = (
        "Classify a CMS run as GOOD, UNCERTAIN, or BAD from a "
        "pre-computed feature vector using a trained ML classifier."
    )

    RuntimeFields = ClassifierRuntimeFields
    StateFields = ClassifierStateFields

    def __init__(self) -> None:
        self._model = None  # loaded lazily
        self._label_map = {0: RunLabel.BAD, 1: RunLabel.UNCERTAIN, 2: RunLabel.GOOD}

    def _load_model(self, checkpoint_path: Optional[str]) -> None:
        """Load a scikit-learn or XGBoost classifier from a pickle file."""
        # TODO: implement loading, e.g.:
        # import joblib
        # self._model = joblib.load(checkpoint_path)
        self._model = None  # stub

    def execute(
        self,
        runtime: ClassifierRuntimeFields,
        state: ClassifierStateFields,
    ) -> dict:
        """Predict run quality from a feature vector.

        Parameters
        ----------
        runtime:
            Feature vector and optional checkpoint path.
        state:
            Updated with prediction and probabilities.

        Returns
        -------
        dict
            Keys: label (str), probabilities (dict[str, float]),
            confidence (float).
        """
        if self._model is None or runtime.model_checkpoint:
            self._load_model(runtime.model_checkpoint)

        # TODO: replace with actual model inference:
        # X = np.array(runtime.feature_vector).reshape(1, -1)
        # proba = self._model.predict_proba(X)[0]
        # idx = int(np.argmax(proba))

        # Stub: deterministic placeholder
        proba = np.array([0.05, 0.10, 0.85])
        idx = int(np.argmax(proba))
        label = self._label_map[idx]

        probabilities = {
            RunLabel.BAD: float(proba[0]),
            RunLabel.UNCERTAIN: float(proba[1]),
            RunLabel.GOOD: float(proba[2]),
        }

        state.last_prediction = label
        state.last_probabilities = probabilities

        return {
            "label": label,
            "probabilities": probabilities,
            "confidence": float(proba[idx]),
        }
