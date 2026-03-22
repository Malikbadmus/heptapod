"""
heptapod.tools.dqm.data.run_registry
=====================================
Tool for querying the CMS Run Registry.

The Run Registry is the authoritative store of per-run certification flags,
detector conditions, and data-taking metadata. This tool wraps the
RunRegistry REST API so the agent can retrieve run metadata before deciding
which histograms to fetch or which reference run to use.

References
----------
* https://cms-run-registry.web.cern.ch
* https://github.com/cms-sw/cmssw/tree/master/CondFormats/RunInfo
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Runtime / State field schemas
# ---------------------------------------------------------------------------


class RunRegistryRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    run_number: int = Field(
        ...,
        description="CMS run number to look up.",
        ge=1,
    )
    dataset: str = Field(
        default="/PromptReco/Run2024*/DQM",
        description="DQM dataset path pattern to match against.",
    )
    include_lumisections: bool = Field(
        default=False,
        description="If True, also return per-LS certification flags.",
    )


class RunRegistryStateFields(BaseModel):
    """Agent state updated after each invocation."""

    last_queried_run: Optional[int] = None
    cached_run_info: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class RunRegistryTool:
    """Query the CMS Run Registry for run metadata and certification status.

    Returns a structured summary of detector conditions, trigger menu,
    data-taking start/end times, and the nominal DQM certification flag
    (GOOD / BAD / NOTSET) for the requested run.

    The agent uses this information to:
      1. Confirm the run exists and was recorded with the expected conditions.
      2. Select an appropriate reference run for comparison.
      3. Decide whether to escalate to a shifter alert.

    Usage
    -----
    >>> tool = RunRegistryTool()
    >>> result = tool.execute(
    ...     RunRegistryRuntimeFields(run_number=380238),
    ...     RunRegistryStateFields(),
    ... )
    """

    name: str = "run_registry_query"
    description: str = (
        "Query the CMS Run Registry to retrieve run metadata, detector "
        "conditions, and certification flags for a given run number."
    )

    # Runtime / State types exposed for JSON-schema generation
    RuntimeFields = RunRegistryRuntimeFields
    StateFields = RunRegistryStateFields

    def execute(
        self,
        runtime: RunRegistryRuntimeFields,
        state: RunRegistryStateFields,
    ) -> dict:
        """Fetch run metadata from the CMS Run Registry REST API.

        Parameters
        ----------
        runtime:
            Per-invocation inputs (run number, dataset, flags).
        state:
            Mutable agent state; updated with the queried run on success.

        Returns
        -------
        dict
            Keys: run_number, certification, start_time, end_time,
            n_lumisections, trigger_menu, conditions_tag,
            and optionally ls_flags (list).

        Raises
        ------
        RuntimeError
            If the Run Registry API is unreachable or returns a non-200 status.
        ValueError
            If the run number does not exist in the registry.
        """
        # TODO: replace stub with live RunRegistry REST call, e.g.
        #   import requests
        #   resp = requests.get(
        #       f"https://cms-run-registry.web.cern.ch/api/runs/{runtime.run_number}",
        #       params={"dataset": runtime.dataset},
        #       cert=("/path/to/cert.pem", "/path/to/key.pem"),
        #   )
        #   resp.raise_for_status()
        #   payload = resp.json()

        state.last_queried_run = runtime.run_number
        state.cached_run_info[runtime.run_number] = {"status": "stub"}

        return {
            "run_number": runtime.run_number,
            "certification": "NOTSET",   # stub
            "n_lumisections": 0,         # stub
            "trigger_menu": "unknown",   # stub
            "conditions_tag": "unknown", # stub
            "ls_flags": [] if runtime.include_lumisections else None,
        }
