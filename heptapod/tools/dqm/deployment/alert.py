"""
heptapod.tools.dqm.deployment.alert
======================================
Tool for dispatching structured alerts to CMS DQM shifters.

Human oversight is a core HEPTAPOD principle: the agent never autonomously
stops a run or certifies data as BAD. Instead it raises structured alerts
so a shifter can make the final decision.

Alert channels
--------------
* mattermost  – CMS ops Mattermost (preferred for online shifts)
* email       – CERN mailing lists (preferred for offline summaries)
* logbook     – CERN e-logbook entry (permanent record)
* stdout      – local console (development / dry-run)

Every alert includes a structured payload so the shifter sees:
  run / LS, subsystem, anomaly score, flagged histograms, recommendation,
  and a direct link to the DQM GUI.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from heptapod.tools.dqm.deployment.run_card_dqm import AlertSeverity


class ShifterAlertRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    run_number: int = Field(..., ge=1)
    lumisection: Optional[int] = Field(default=None)
    subsystem: str = Field(...)
    severity: AlertSeverity = Field(default=AlertSeverity.WARNING)
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    flagged_histograms: list[str] = Field(default_factory=list)
    recommendation: str = Field(
        ...,
        description="GO | INVESTIGATE | STOP",
    )
    message: Optional[str] = Field(
        default=None,
        description="Optional human-readable summary appended to the alert.",
    )
    channel: str = Field(
        default="stdout",
        description="Alert channel: 'mattermost', 'email', 'logbook', 'stdout'.",
    )
    dry_run: bool = Field(
        default=False,
        description="If True, format the alert but do not dispatch it.",
    )


class ShifterAlertStateFields(BaseModel):
    """Agent state updated after each invocation."""

    alerts_dispatched: int = 0
    last_alert_time: Optional[str] = None
    alert_log: list[dict] = Field(default_factory=list)


class ShifterAlertTool:
    """Dispatch a structured alert to CMS DQM shifters.

    Preserves the human-in-the-loop requirement: the agent proposes a
    recommendation but the shifter retains full authority to accept or
    override it.

    Alert payload (always included)
    --------------------------------
    * Run / LS identifier
    * Subsystem and severity
    * Anomaly score and flagged histogram list
    * Recommendation (GO / INVESTIGATE / STOP)
    * DQM GUI deep-link for the run
    * UTC timestamp

    Usage
    -----
    >>> tool = ShifterAlertTool()
    >>> tool.execute(
    ...     ShifterAlertRuntimeFields(
    ...         run_number=380238, subsystem="ecal",
    ...         anomaly_score=0.93, recommendation="INVESTIGATE",
    ...         flagged_histograms=["EBOcc", "EBPed"],
    ...         severity=AlertSeverity.WARNING,
    ...     ),
    ...     ShifterAlertStateFields(),
    ... )
    """

    name: str = "shifter_alert"
    description: str = (
        "Dispatch a structured DQM alert to CMS shifters via the configured "
        "channel (Mattermost, email, logbook, or stdout)."
    )

    RuntimeFields = ShifterAlertRuntimeFields
    StateFields = ShifterAlertStateFields

    _DQM_GUI_BASE = "https://cmsweb.cern.ch/dqm/offline/#run={run};dataset={dataset}"

    def _format_payload(self, runtime: ShifterAlertRuntimeFields) -> str:
        ls_str = f" LS {runtime.lumisection}" if runtime.lumisection else ""
        gui_link = self._DQM_GUI_BASE.format(
            run=runtime.run_number, dataset="PromptReco"
        )
        flagged_str = (
            ", ".join(runtime.flagged_histograms)
            if runtime.flagged_histograms
            else "none"
        )
        return (
            f"[ML4DQM {runtime.severity.upper()}] "
            f"Run {runtime.run_number}{ls_str} | {runtime.subsystem.upper()}\n"
            f"  Anomaly score : {runtime.anomaly_score:.3f}\n"
            f"  Flagged MEs   : {flagged_str}\n"
            f"  Recommendation: {runtime.recommendation}\n"
            f"  DQM GUI       : {gui_link}\n"
            + (f"  Note          : {runtime.message}\n" if runtime.message else "")
        )

    def execute(
        self,
        runtime: ShifterAlertRuntimeFields,
        state: ShifterAlertStateFields,
    ) -> dict:
        """Format and dispatch a shifter alert.

        Parameters
        ----------
        runtime:
            Alert content and dispatch channel.
        state:
            Updated with dispatch count, timestamp, and log.

        Returns
        -------
        dict
            Keys: dispatched (bool), channel (str), payload (str),
            timestamp (str).
        """
        payload = self._format_payload(runtime)
        timestamp = datetime.now(timezone.utc).isoformat()

        if not runtime.dry_run:
            if runtime.channel == "stdout":
                print(payload)

            elif runtime.channel == "mattermost":
                # TODO: implement Mattermost webhook POST
                # import requests
                # requests.post(MATTERMOST_WEBHOOK_URL,
                #               json={"text": payload})
                pass

            elif runtime.channel == "email":
                # TODO: implement CERN SMTP send
                pass

            elif runtime.channel == "logbook":
                # TODO: implement CERN e-logbook API call
                pass

        state.alerts_dispatched += 1
        state.last_alert_time = timestamp
        state.alert_log.append({
            "run": runtime.run_number,
            "severity": runtime.severity,
            "recommendation": runtime.recommendation,
            "timestamp": timestamp,
            "dry_run": runtime.dry_run,
        })

        return {
            "dispatched": not runtime.dry_run,
            "channel": runtime.channel,
            "payload": payload,
            "timestamp": timestamp,
        }
