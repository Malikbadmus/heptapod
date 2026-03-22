"""
heptapod.tools.dqm.deployment.monitor
========================================
Tool for real-time online monitoring during CMS data-taking.

In online DQM mode the agent is invoked per-lumisection (~23 seconds) to
check whether the detector is operating within normal bounds. This tool
wraps the entry point for that loop: it fetches the latest histograms,
scores them, and returns a go/no-go recommendation to the orchestrator.

Latency budget
--------------
Online DQM runs within a ~5-second window per lumisection:
  * Histogram fetch  : ~1–2 s  (DQMGuiFetchTool, local socket)
  * Feature extract  : ~0.1 s  (HistogramFeatureTool)
  * AE inference     : ~0.1 s  (AutoencoderAnomalyTool, ONNX Runtime)
  * Classification   : ~0.05 s (RunQualityClassifierTool)
  * Alert dispatch   : ~0.1 s  (ShifterAlertTool)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class OnlineMonitorRuntimeFields(BaseModel):
    """Inputs consumed per invocation (one per lumisection)."""

    run_number: int = Field(..., description="Current CMS run number.", ge=1)
    lumisection: int = Field(..., description="Current lumisection number.", ge=1)
    subsystem: str = Field(..., description="Sub-detector subsystem to monitor.")
    dqm_socket_url: str = Field(
        default="http://localhost:9090/dqm",
        description="URL of the local DQM GUI socket (online mode).",
    )
    model_checkpoint: Optional[str] = Field(
        default=None,
        description="Model checkpoint override. None = use config default.",
    )


class OnlineMonitorStateFields(BaseModel):
    """Agent state updated after each lumisection."""

    current_run: Optional[int] = None
    n_lumisections_processed: int = 0
    n_alerts_raised: int = 0
    consecutive_bad_ls: int = 0


class OnlineMonitorTool:
    """Run one step of the online DQM monitoring loop.

    Intended to be called in a loop by the orchestrator during data-taking:

        while data_taking:
            result = monitor.execute(runtime, state)
            if result["recommendation"] == "STOP":
                orchestrator.pause_run()

    The tool is stateful: it tracks how many consecutive lumisections have
    been flagged and escalates alerts accordingly.

    Usage
    -----
    >>> tool = OnlineMonitorTool()
    >>> result = tool.execute(
    ...     OnlineMonitorRuntimeFields(
    ...         run_number=380238, lumisection=42, subsystem="ecal"
    ...     ),
    ...     OnlineMonitorStateFields(current_run=380238),
    ... )
    """

    name: str = "online_monitor"
    description: str = (
        "Execute one step of the online DQM monitoring loop for a single "
        "lumisection and return a go/no-go recommendation."
    )

    RuntimeFields = OnlineMonitorRuntimeFields
    StateFields = OnlineMonitorStateFields

    # Number of consecutive bad LS before recommending a run stop
    CONSECUTIVE_BAD_THRESHOLD: int = 5

    def execute(
        self,
        runtime: OnlineMonitorRuntimeFields,
        state: OnlineMonitorStateFields,
    ) -> dict:
        """Monitor a single lumisection and return a recommendation.

        Parameters
        ----------
        runtime:
            Run number, lumisection, subsystem, and socket URL.
        state:
            Updated with processed LS count and consecutive bad LS counter.

        Returns
        -------
        dict
            Keys: recommendation ("GO" | "INVESTIGATE" | "STOP"),
            anomaly_score (float), n_flagged_histograms (int),
            lumisection (int), run_number (int).
        """
        # The full online pipeline is:
        # 1. DQMGuiFetchTool  → raw histograms
        # 2. HistogramFeatureTool → feature vector
        # 3. AutoencoderAnomalyTool + ReferenceCompareTool → scores
        # 4. RunQualityClassifierTool → label
        # 5. Aggregate into a recommendation

        # TODO: wire up the sub-tools here via the orchestrator's tool registry
        # For now, return a stub recommendation.

        state.current_run = runtime.run_number
        state.n_lumisections_processed += 1

        # Stub: always GO
        anomaly_score = 0.05
        n_flagged = 0
        is_bad = anomaly_score > 0.80

        if is_bad:
            state.consecutive_bad_ls += 1
            state.n_alerts_raised += 1
        else:
            state.consecutive_bad_ls = 0

        if state.consecutive_bad_ls >= self.CONSECUTIVE_BAD_THRESHOLD:
            recommendation = "STOP"
        elif is_bad:
            recommendation = "INVESTIGATE"
        else:
            recommendation = "GO"

        return {
            "recommendation": recommendation,
            "anomaly_score": anomaly_score,
            "n_flagged_histograms": n_flagged,
            "lumisection": runtime.lumisection,
            "run_number": runtime.run_number,
            "consecutive_bad_ls": state.consecutive_bad_ls,
        }
