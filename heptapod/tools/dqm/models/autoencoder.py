"""
heptapod.tools.dqm.models.autoencoder
========================================
Tool for autoencoder-based histogram anomaly detection.

An autoencoder trained on known-good (certified GOOD) runs learns a compact
representation of normal detector behaviour. Anomalous histograms — caused by
dead channels, noisy ASICs, beam instabilities, or DAQ errors — produce high
reconstruction errors and are flagged as outliers.

Architecture (default)
----------------------
Encoder: Linear(n_bins → 128) → ReLU → Linear(128 → 32) → ReLU
Bottleneck: Linear(32 → latent_dim)
Decoder: mirror of encoder, final Sigmoid

The reconstruction error is the mean-squared error between input and output
histogram bin values (after normalisation to [0, 1]).

References
----------
* Pol et al., "Anomaly detection with conditional variational autoencoders"
  arXiv:1911.10461
* CMS DQM ML group: https://cms-ml.github.io/documentation/
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class AutoencoderRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    histogram_values: list[list[float]] = Field(
        ...,
        description=(
            "Batch of normalised histogram bin values. "
            "Shape: (n_histograms, n_bins). Flattened for TH2."
        ),
    )
    model_checkpoint: Optional[str] = Field(
        default=None,
        description="Path to a saved PyTorch state_dict. None = use loaded model.",
    )
    return_latent: bool = Field(
        default=False,
        description="If True, also return the bottleneck latent vectors.",
    )


class AutoencoderStateFields(BaseModel):
    """Agent state updated after each invocation."""

    model_loaded: bool = False
    last_reconstruction_errors: list[float] = Field(default_factory=list)
    anomaly_flags: list[bool] = Field(default_factory=list)


class AutoencoderAnomalyTool:
    """Detect anomalous CMS DQM histograms with a trained autoencoder.

    Scores each histogram in a batch by its reconstruction error and returns
    a flag (True = anomalous) based on a configurable percentile threshold
    set in the DQMRunCard.

    The agent uses this tool as the primary anomaly detector:
      1. Fetch histograms with DQMGuiFetchTool or RootHistogramTool.
      2. Preprocess with HistogramFeatureTool.
      3. Score with AutoencoderAnomalyTool.
      4. Escalate flagged histograms via ShifterAlertTool.

    Usage
    -----
    >>> tool = AutoencoderAnomalyTool()
    >>> result = tool.execute(
    ...     AutoencoderRuntimeFields(histogram_values=[[0.1, 0.9, ...]]),
    ...     AutoencoderStateFields(),
    ... )
    """

    name: str = "autoencoder_anomaly_detection"
    description: str = (
        "Score a batch of DQM histograms with a trained autoencoder and "
        "return per-histogram reconstruction errors and anomaly flags."
    )

    RuntimeFields = AutoencoderRuntimeFields
    StateFields = AutoencoderStateFields

    def __init__(self, anomaly_threshold_percentile: float = 95.0) -> None:
        """
        Parameters
        ----------
        anomaly_threshold_percentile:
            Reconstruction errors above this percentile of the training
            distribution are labelled anomalous.
        """
        self.anomaly_threshold_percentile = anomaly_threshold_percentile
        self._model = None  # loaded lazily

    def _load_model(self, checkpoint_path: Optional[str]) -> None:
        """Load (or reload) a PyTorch autoencoder from a checkpoint."""
        try:
            import torch  # type: ignore
            from torch import nn  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for AutoencoderAnomalyTool. "
                "Install with: pip install torch"
            ) from exc

        # TODO: replace with actual model class from ml4dqm_models package
        # from heptapod.tools.dqm.models._architectures import DQMAutoencoder
        # self._model = DQMAutoencoder(...)
        # if checkpoint_path:
        #     self._model.load_state_dict(torch.load(checkpoint_path))
        # self._model.eval()
        self._model = None  # stub

    def execute(
        self,
        runtime: AutoencoderRuntimeFields,
        state: AutoencoderStateFields,
    ) -> dict:
        """Score histograms and return reconstruction errors and anomaly flags.

        Parameters
        ----------
        runtime:
            Batch of normalised histogram bin values + optional checkpoint.
        state:
            Updated with per-run error and flag lists.

        Returns
        -------
        dict
            Keys: reconstruction_errors (list[float]),
                  anomaly_flags (list[bool]),
                  latent_vectors (list[list[float]] | None).
        """
        if self._model is None or runtime.model_checkpoint:
            self._load_model(runtime.model_checkpoint)

        X = np.array(runtime.histogram_values, dtype=np.float32)

        # TODO: replace stub inference with actual forward pass:
        # import torch
        # with torch.no_grad():
        #     X_t = torch.tensor(X)
        #     X_hat = self._model(X_t).numpy()
        # reconstruction_errors = np.mean((X - X_hat) ** 2, axis=1).tolist()

        # Stub: random errors for structure demonstration
        reconstruction_errors = np.random.rand(len(X)).tolist()
        threshold = np.percentile(reconstruction_errors, self.anomaly_threshold_percentile)
        anomaly_flags = [e > threshold for e in reconstruction_errors]

        state.last_reconstruction_errors = reconstruction_errors
        state.anomaly_flags = anomaly_flags
        state.model_loaded = self._model is not None

        return {
            "reconstruction_errors": reconstruction_errors,
            "anomaly_flags": anomaly_flags,
            "threshold_used": float(threshold),
            "latent_vectors": None,  # populated when return_latent=True
        }
