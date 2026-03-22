"""
heptapod.tools.dqm.deployment.run_card_dqm
===========================================
DQM-specific run-card schema.

Run cards are HEPTAPOD's canonical orchestration boundary: a structured,
versionable config object the agent writes and submits to external tools.
Every DQM action is parameterised by a DQMRunCard so that the full decision
trace is auditable and reproducible.

A DQMRunCard captures:
  - which CMS run (or run range) is being assessed
  - which sub-detector subsystem is in scope
  - which reference run to compare against
  - operating mode (online / offline)
  - model checkpoint to use for inference
  - alert thresholds
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DQMMode(str, Enum):
    """Operating mode for the DQM agent.

    ONLINE  – real-time monitoring during data-taking; latency is critical.
    OFFLINE – post-run reprocessing; thoroughness takes priority.
    """
    ONLINE = "online"
    OFFLINE = "offline"


class Subsystem(str, Enum):
    """CMS sub-detector subsystems supported by ML4DQM."""
    TRACKER = "tracker"
    ECAL = "ecal"
    HCAL = "hcal"
    MUON = "muon"
    GLOBAL = "global"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class DQMRunCard(BaseModel):
    """Structured configuration for a single DQM agent invocation.

    The run card is written by the agent before calling any DQM tool and
    stored alongside every tool result for full reproducibility.

    Example
    -------
    >>> card = DQMRunCard(
    ...     run_number=380238,
    ...     subsystem=Subsystem.ECAL,
    ...     reference_run=379100,
    ...     mode=DQMMode.OFFLINE,
    ... )
    """

    # -- Run identification ------------------------------------------------
    run_number: int = Field(
        ...,
        description="CMS run number to be assessed.",
        ge=1,
    )
    lumisection_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="Inclusive (start, end) lumisection range. None = all LS.",
    )

    # -- Scope -------------------------------------------------------------
    subsystem: Subsystem = Field(
        ...,
        description="Sub-detector subsystem to monitor.",
    )
    histogram_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit DQM histogram paths to fetch. "
            "If empty the tool uses the subsystem default set."
        ),
    )

    # -- Reference ---------------------------------------------------------
    reference_run: Optional[int] = Field(
        default=None,
        description=(
            "Run number of a known-good reference run for comparison. "
            "If None, the model uses its training distribution baseline."
        ),
    )

    # -- Model -------------------------------------------------------------
    model_checkpoint: Optional[str] = Field(
        default=None,
        description="Path or registry key of the ML model checkpoint to use.",
    )
    anomaly_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Reconstruction-error percentile above which a histogram is flagged.",
    )

    # -- Operation ---------------------------------------------------------
    mode: DQMMode = Field(
        default=DQMMode.OFFLINE,
        description="Online (real-time) or offline (post-run) assessment.",
    )
    alert_severity_floor: AlertSeverity = Field(
        default=AlertSeverity.WARNING,
        description="Minimum severity level that triggers a shifter alert.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, produce outputs but do not emit alerts or write to registry.",
    )

    class Config:
        use_enum_values = True
