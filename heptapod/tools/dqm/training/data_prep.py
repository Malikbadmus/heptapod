"""
heptapod.tools.dqm.training.data_prep
========================================
Tool for extracting ML-ready feature vectors from DQM histograms.

Raw histogram bin values are high-dimensional (1024+ bins for TH2) and
noisy. This tool computes a compact, physically motivated feature set
that is stable across different luminosity conditions.

Feature groups (configurable)
------------------------------
summary_stats   – mean, std, skewness, kurtosis, n_entries
occupancy       – fraction of empty bins, fraction of hot bins (> threshold)
shape           – first 10 PCA components of normalised bin values
ae_scores       – reconstruction errors from the autoencoder (if provided)
ref_delta       – p-values from reference comparison (if provided)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import kurtosis, skew  # type: ignore


class HistogramFeatureRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    histogram_values: dict[str, list[float]] = Field(
        ...,
        description="Mapping of histogram name → normalised bin values.",
    )
    ae_reconstruction_errors: Optional[dict[str, float]] = Field(
        default=None,
        description="Autoencoder reconstruction errors keyed by histogram name.",
    )
    reference_p_values: Optional[dict[str, float]] = Field(
        default=None,
        description="Reference-comparison p-values keyed by histogram name.",
    )
    hot_bin_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Bin fraction above which a bin is considered 'hot'.",
    )
    include_pca: bool = Field(
        default=True,
        description="Append first 10 PCA components to the feature vector.",
    )


class HistogramFeatureStateFields(BaseModel):
    """Agent state updated after each invocation."""

    feature_dim: Optional[int] = None
    feature_names: list[str] = Field(default_factory=list)


class HistogramFeatureTool:
    """Extract a compact, ML-ready feature vector from DQM histograms.

    The output vector is suitable for direct input to RunQualityClassifierTool
    or for assembling a training dataset with ModelTrainerTool.

    Usage
    -----
    >>> tool = HistogramFeatureTool()
    >>> result = tool.execute(
    ...     HistogramFeatureRuntimeFields(
    ...         histogram_values={"EBOcc": [0.1, 0.2, ...]},
    ...     ),
    ...     HistogramFeatureStateFields(),
    ... )
    """

    name: str = "histogram_feature_extractor"
    description: str = (
        "Extract a compact feature vector from DQM histogram bin values "
        "for use as ML model input."
    )

    RuntimeFields = HistogramFeatureRuntimeFields
    StateFields = HistogramFeatureStateFields

    def execute(
        self,
        runtime: HistogramFeatureRuntimeFields,
        state: HistogramFeatureStateFields,
    ) -> dict:
        """Compute summary statistics and shape features for each histogram.

        Parameters
        ----------
        runtime:
            Histogram bin values and optional AE/reference scores.
        state:
            Updated with output feature dimensionality and names.

        Returns
        -------
        dict
            Keys: feature_vector (list[float]), feature_names (list[str]).
        """
        features: list[float] = []
        names: list[str] = []

        for hist_name, bins in runtime.histogram_values.items():
            arr = np.array(bins, dtype=np.float64)
            n = len(arr)

            # Summary statistics
            features += [
                float(np.mean(arr)),
                float(np.std(arr)),
                float(skew(arr)),
                float(kurtosis(arr)),
                float(np.sum(arr)),  # proxy for n_entries if normalised
            ]
            names += [
                f"{hist_name}_mean", f"{hist_name}_std",
                f"{hist_name}_skew", f"{hist_name}_kurt",
                f"{hist_name}_sum",
            ]

            # Occupancy
            empty_frac = float(np.mean(arr == 0.0))
            hot_frac = float(np.mean(arr > runtime.hot_bin_threshold))
            features += [empty_frac, hot_frac]
            names += [f"{hist_name}_empty_frac", f"{hist_name}_hot_frac"]

            # PCA shape features
            if runtime.include_pca and n >= 10:
                from sklearn.decomposition import PCA  # type: ignore
                n_components = min(10, n)
                pca = PCA(n_components=n_components)
                # Treat the single histogram as a (1 x n) matrix; use the
                # row projection as a shape fingerprint
                shape_feats = pca.fit_transform(arr.reshape(1, -1))[0]
                features += shape_feats.tolist()
                names += [f"{hist_name}_pca_{i}" for i in range(n_components)]

            # Autoencoder score
            if runtime.ae_reconstruction_errors and hist_name in runtime.ae_reconstruction_errors:
                features.append(runtime.ae_reconstruction_errors[hist_name])
                names.append(f"{hist_name}_ae_error")

            # Reference p-value
            if runtime.reference_p_values and hist_name in runtime.reference_p_values:
                features.append(runtime.reference_p_values[hist_name])
                names.append(f"{hist_name}_ref_pval")

        state.feature_dim = len(features)
        state.feature_names = names

        return {"feature_vector": features, "feature_names": names}
