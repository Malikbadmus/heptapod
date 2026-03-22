"""
heptapod.tools.dqm.data.root_reader
=====================================
Tool for reading CMS Monitor Elements directly from ROOT files via uproot.

DQM output files produced by CMSSW follow the path convention:
  DQMData/Run {run}/Subsystem/Run summary/...

This tool provides a Python-native interface (no ROOT installation required)
by using the `uproot` library to read TH1 and TH2 histograms from .root files
and convert them to numpy arrays for ML pipelines.

Dependencies
------------
* uproot  (pip install uproot)
* numpy   (pip install numpy)

References
----------
* https://uproot.readthedocs.io
* https://cms-dqm.web.cern.ch/cms-dqm/getting-started/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


class RootHistogramRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    root_file_path: str = Field(
        ...,
        description="Absolute or relative path to the DQM .root file.",
    )
    histogram_keys: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of histogram keys inside the ROOT file, e.g. "
            "['DQMData/Run 380238/EcalBarrel/Run summary/"
            "EBOccupancyTask/EBOT digi occupancy']."
        ),
    )
    normalise: bool = Field(
        default=True,
        description="Divide bin values by the total number of entries.",
    )


class RootHistogramStateFields(BaseModel):
    """Agent state updated after each invocation."""

    loaded_files: list[str] = Field(default_factory=list)


class RootHistogramTool:
    """Read CMS DQM histograms from .root files using uproot.

    Converts TH1F / TH2F objects to normalised numpy arrays that can be
    passed directly into ML model tools without a ROOT installation.

    The agent uses this tool when:
      - DQM data is available as local .root files (offline batch jobs).
      - The DQM GUI API is unavailable or the run is not yet ingested.

    Usage
    -----
    >>> tool = RootHistogramTool()
    >>> result = tool.execute(
    ...     RootHistogramRuntimeFields(
    ...         root_file_path="/data/dqm/DQM_V0001_R000380238.root",
    ...         histogram_keys=["DQMData/Run 380238/EcalBarrel/..."],
    ...     ),
    ...     RootHistogramStateFields(),
    ... )
    """

    name: str = "root_histogram_reader"
    description: str = (
        "Read CMS DQM Monitor Element histograms from a ROOT file and "
        "return them as normalised numpy arrays."
    )

    RuntimeFields = RootHistogramRuntimeFields
    StateFields = RootHistogramStateFields

    def execute(
        self,
        runtime: RootHistogramRuntimeFields,
        state: RootHistogramStateFields,
    ) -> dict[str, dict]:
        """Open a DQM ROOT file and extract the requested histograms.

        Parameters
        ----------
        runtime:
            Path to .root file and list of histogram keys.
        state:
            Tracks which files have been opened in this session.

        Returns
        -------
        dict[str, dict]
            Mapping of key → {"values": np.ndarray, "edges": list[np.ndarray],
            "entries": int, "type": str}.

        Raises
        ------
        FileNotFoundError
            If root_file_path does not exist.
        KeyError
            If a histogram key is absent from the file.
        ImportError
            If uproot is not installed.
        """
        try:
            import uproot  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "uproot is required for RootHistogramTool. "
                "Install with: pip install uproot"
            ) from exc

        path = Path(runtime.root_file_path)
        if not path.exists():
            raise FileNotFoundError(f"ROOT file not found: {path}")

        state.loaded_files.append(str(path))
        results: dict[str, dict] = {}

        with uproot.open(str(path)) as f:
            for key in runtime.histogram_keys:
                hist = f[key]
                values, edges = hist.to_numpy()
                n_entries = int(hist.values().sum())

                if runtime.normalise and n_entries > 0:
                    values = values / n_entries

                results[key] = {
                    "values": values,
                    "edges": edges if isinstance(edges, list) else [edges],
                    "entries": n_entries,
                    "type": type(hist).__name__,
                }

        return results
