"""
heptapod.tools.dqm.models.reference_compare
=============================================
Tool for statistical comparison between a current run and a reference run.

Reference-based DQM is a standard CMS approach: a shifter manually certifies
one run as GOOD; subsequent runs are compared against it using statistical
tests. A large deviation signals a potential detector problem.

Supported tests
---------------
* chi2    – Chi-squared test (ROOT-style normalised chi²/ndf)
* ks      – Kolmogorov-Smirnov test on 1D histograms
* bdm     – Bin-by-bin difference metric (|current - ref| / sqrt(ref))

References
----------
* CMS DQM quality tests: https://cms-dqm.web.cern.ch/cms-dqm/quality-tests/
* Gleyzer et al., "Feature Engineering and Deep Learning", arXiv:2012.04517
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats  # type: ignore


class StatTest(str, Enum):
    CHI2 = "chi2"
    KS = "ks"
    BDM = "bdm"


class ReferenceCompareRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    current_histograms: dict[str, list[float]] = Field(
        ...,
        description="Mapping of histogram name → bin values for the current run.",
    )
    reference_histograms: dict[str, list[float]] = Field(
        ...,
        description="Mapping of histogram name → bin values for the reference run.",
    )
    stat_test: StatTest = Field(
        default=StatTest.KS,
        description="Statistical test to apply.",
    )
    p_value_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="p-value below which a histogram is flagged as anomalous.",
    )


class ReferenceCompareStateFields(BaseModel):
    """Agent state updated after each invocation."""

    last_p_values: dict[str, float] = Field(default_factory=dict)
    flagged_histograms: list[str] = Field(default_factory=list)


class ReferenceCompareTool:
    """Compare current-run histograms against a certified reference run.

    Returns per-histogram p-values and a list of histograms that fail the
    chosen statistical test at the configured significance level.

    Complements the autoencoder: the autoencoder detects unusual *shapes*
    globally; reference comparison detects *local* deviations from a known
    baseline and is more interpretable to shifters.

    Usage
    -----
    >>> tool = ReferenceCompareTool()
    >>> result = tool.execute(
    ...     ReferenceCompareRuntimeFields(
    ...         current_histograms={"EBOcc": [0.1, 0.2, ...]},
    ...         reference_histograms={"EBOcc": [0.1, 0.19, ...]},
    ...     ),
    ...     ReferenceCompareStateFields(),
    ... )
    """

    name: str = "reference_compare"
    description: str = (
        "Statistically compare current-run DQM histograms against a "
        "reference run and flag histograms that deviate significantly."
    )

    RuntimeFields = ReferenceCompareRuntimeFields
    StateFields = ReferenceCompareStateFields

    def execute(
        self,
        runtime: ReferenceCompareRuntimeFields,
        state: ReferenceCompareStateFields,
    ) -> dict:
        """Run statistical tests and return per-histogram scores.

        Parameters
        ----------
        runtime:
            Current and reference histogram dicts, test choice, threshold.
        state:
            Updated with p-values and flagged histogram list.

        Returns
        -------
        dict
            Keys: p_values (dict[str, float]), flagged (list[str]),
            test_used (str).
        """
        p_values: dict[str, float] = {}
        common_keys = set(runtime.current_histograms) & set(runtime.reference_histograms)

        for name in common_keys:
            cur = np.array(runtime.current_histograms[name], dtype=np.float64)
            ref = np.array(runtime.reference_histograms[name], dtype=np.float64)

            if runtime.stat_test == StatTest.KS:
                # Convert to pseudo-distributions by repeating indices
                _, p = stats.ks_2samp(cur, ref)

            elif runtime.stat_test == StatTest.CHI2:
                # Avoid zero-division; add small epsilon
                expected = ref + 1e-9
                chi2_stat = np.sum((cur - expected) ** 2 / expected)
                dof = max(len(cur) - 1, 1)
                p = float(stats.chi2.sf(chi2_stat, df=dof))

            elif runtime.stat_test == StatTest.BDM:
                # Bin-by-bin difference metric (lower = more similar)
                with np.errstate(divide="ignore", invalid="ignore"):
                    bdm = np.nanmean(np.abs(cur - ref) / np.sqrt(ref + 1e-9))
                # Map to pseudo-p-value: 1 - tanh(bdm) in (0,1]
                p = float(1.0 - np.tanh(bdm))

            else:
                raise ValueError(f"Unknown stat_test: {runtime.stat_test}")

            p_values[name] = float(p)

        flagged = [k for k, p in p_values.items() if p < runtime.p_value_threshold]

        state.last_p_values = p_values
        state.flagged_histograms = flagged

        return {
            "p_values": p_values,
            "flagged": flagged,
            "test_used": runtime.stat_test,
        }
