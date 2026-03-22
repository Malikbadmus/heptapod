"""
heptapod.tools.dqm.data.dqm_gui
================================
Tool for fetching Monitor Element histograms from the CMS DQM GUI.

The CMS DQM GUI (https://cmsweb.cern.ch/dqm/offline/) exposes a REST/JSON
API that returns histogram data as JSON objects. This tool wraps that API
so the agent can pull named histograms without understanding the underlying
HTTP layer.

Fetched histograms are returned as plain Python dicts whose structure
mirrors the DQM GUI JSON schema:
  { "name": str, "type": "TH1F"|"TH2F"|..., "entries": int,
    "xbins": [...], "values": [[...]], "errors": [[...]] }

References
----------
* https://cmsweb.cern.ch/dqm/offline/jsonfairy/
* https://github.com/cms-dqm/dqmgui
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class DQMGuiRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    run_number: int = Field(..., description="CMS run number.", ge=1)
    dataset: str = Field(
        ...,
        description=(
            "Full DQM dataset path, e.g. "
            "'/PromptReco/Run2024G/DQM'."
        ),
    )
    histogram_paths: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of histogram paths relative to the dataset root, e.g. "
            "['EcalBarrel/EBOccupancyTask/EBOT digi occupancy']."
        ),
    )
    lumisection: Optional[int] = Field(
        default=None,
        description="Specific lumisection to fetch. None = run-level summary.",
    )


class DQMGuiStateFields(BaseModel):
    """Agent state updated after each invocation."""

    fetched_histograms: dict[str, dict] = Field(
        default_factory=dict,
        description="Cache keyed by (run, path) string.",
    )


class DQMGuiFetchTool:
    """Fetch detector histograms from the CMS DQM GUI JSON API.

    Returns histogram data as structured dicts ready for downstream
    feature extraction or direct model inference.

    The agent uses this tool to:
      1. Retrieve the specific Monitor Elements flagged in a DQMRunCard.
      2. Pull reference histograms for a known-good run.
      3. Hydrate training datasets for model (re)training.

    Usage
    -----
    >>> tool = DQMGuiFetchTool()
    >>> result = tool.execute(
    ...     DQMGuiRuntimeFields(
    ...         run_number=380238,
    ...         dataset="/PromptReco/Run2024G/DQM",
    ...         histogram_paths=["EcalBarrel/EBOccupancyTask/EBOT digi occupancy"],
    ...     ),
    ...     DQMGuiStateFields(),
    ... )
    """

    name: str = "dqm_gui_fetch"
    description: str = (
        "Fetch one or more Monitor Element histograms from the CMS DQM GUI "
        "for a specified run and dataset."
    )

    RuntimeFields = DQMGuiRuntimeFields
    StateFields = DQMGuiStateFields

    def execute(
        self,
        runtime: DQMGuiRuntimeFields,
        state: DQMGuiStateFields,
    ) -> dict[str, dict]:
        """Retrieve histograms via the DQM GUI JSON fairy API.

        Parameters
        ----------
        runtime:
            Run number, dataset, and histogram paths to fetch.
        state:
            Mutable cache; populated with fetched histogram dicts.

        Returns
        -------
        dict[str, dict]
            Mapping of histogram path → DQM JSON histogram object.

        Raises
        ------
        RuntimeError
            On HTTP errors or CERN SSO authentication failure.
        KeyError
            If a requested histogram path does not exist for the given run.
        """
        # TODO: implement live fetch, e.g.:
        #   import requests
        #   base = "https://cmsweb.cern.ch/dqm/offline/jsonfairy/archive"
        #   for path in runtime.histogram_paths:
        #       url = f"{base}/{runtime.run_number}/{runtime.dataset}/{path}"
        #       r = requests.get(url, cert=CERN_CERT, verify=CERN_CA)
        #       r.raise_for_status()
        #       state.fetched_histograms[f"{runtime.run_number}:{path}"] = r.json()

        results: dict[str, dict] = {}
        for path in runtime.histogram_paths:
            stub = {"name": path, "type": "TH1F", "entries": 0,
                    "xbins": [], "values": [], "errors": []}
            state.fetched_histograms[f"{runtime.run_number}:{path}"] = stub
            results[path] = stub

        return results
