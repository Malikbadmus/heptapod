"""
examples/dqm_demo.py
====================
End-to-end demo of the ML4DQM HEPTAPOD agent.

Mirrors hep_bsm_demo.py but targets CMS Data Quality Monitoring instead of
BSM event generation. The agent is given a natural-language DQM task and
orchestrates the full pipeline:

    Run Registry query → histogram fetch → feature extraction →
    autoencoder scoring → reference comparison → classification →
    shifter alert

Usage
-----
    python examples/dqm_demo.py --run 380238 --subsystem ecal --mode offline

Dependencies
------------
    pip install heptapod[dqm]   # once packaging is complete
    # or from the repo root:
    pip install -e ".[dqm]"
"""

from __future__ import annotations

import argparse
import json

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML4DQM HEPTAPOD agent demo"
    )
    parser.add_argument(
        "--run", type=int, default=380238,
        help="CMS run number to assess (default: 380238)",
    )
    parser.add_argument(
        "--subsystem", default="ecal",
        choices=["tracker", "ecal", "hcal", "muon", "global"],
        help="Sub-detector subsystem (default: ecal)",
    )
    parser.add_argument(
        "--mode", default="offline",
        choices=["online", "offline"],
        help="DQM operating mode (default: offline)",
    )
    parser.add_argument(
        "--reference-run", type=int, default=None,
        help="Reference run number (default: auto-select from registry)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run pipeline but do not emit real alerts",
    )
    parser.add_argument(
        "--llm-backend", default="ollama",
        choices=["ollama", "anthropic", "openai"],
        help="LLM backend for the orchestrator (default: ollama)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Agent task
# ---------------------------------------------------------------------------

TASK_TEMPLATE = """
You are an ML4DQM agent responsible for assessing the data quality of
CMS Run {run_number} ({subsystem} subsystem, {mode} mode).

Your goal is to:
1. Query the Run Registry to confirm the run exists and retrieve its metadata.
2. Fetch the standard set of {subsystem} Monitor Element histograms.
3. Compare histograms against reference run {reference_run} using the
   reference_compare tool (KS test, p < 0.05).
4. Score histograms with the autoencoder_anomaly_detection tool.
5. Extract features with histogram_feature_extractor.
6. Classify the overall run quality with run_quality_classifier.
7. If the classification is BAD or UNCERTAIN, raise a shifter alert via
   the shifter_alert tool with the appropriate severity.
8. Summarise your findings in a structured DQMRunCard.

Use only the available tools. Do not make assumptions about detector
conditions not supported by the tool outputs. If uncertain, escalate to
a human shifter rather than making a unilateral decision.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -- Resolve reference run ----------------------------------------------
    reference_run = args.reference_run or (args.run - 100)  # simple heuristic

    task = TASK_TEMPLATE.format(
        run_number=args.run,
        subsystem=args.subsystem,
        mode=args.mode,
        reference_run=reference_run,
    )

    # -- Build DQMRunCard ---------------------------------------------------
    from heptapod.tools.dqm.deployment.run_card_dqm import (
        DQMRunCard, DQMMode, Subsystem
    )

    run_card = DQMRunCard(
        run_number=args.run,
        subsystem=Subsystem(args.subsystem),
        reference_run=reference_run,
        mode=DQMMode(args.mode),
        dry_run=args.dry_run,
    )
    print("=== DQM Run Card ===")
    print(json.dumps(run_card.dict(), indent=2, default=str))

    # -- Instantiate tools --------------------------------------------------
    from heptapod.tools.dqm import (
        RunRegistryTool, DQMGuiFetchTool,
        AutoencoderAnomalyTool, ReferenceCompareTool,
        RunQualityClassifierTool, HistogramFeatureTool,
        ShifterAlertTool,
    )

    tools = [
        RunRegistryTool(),
        DQMGuiFetchTool(),
        AutoencoderAnomalyTool(),
        ReferenceCompareTool(),
        HistogramFeatureTool(),
        RunQualityClassifierTool(),
        ShifterAlertTool(),
    ]

    # -- Launch orchestrator ------------------------------------------------
    # TODO: wire up to the HEPTAPOD Orchestral AI engine, e.g.:
    #
    #   from orchestral import Orchestrator
    #   from heptapod.llm.utils import get_llm_client
    #
    #   llm = get_llm_client(backend=args.llm_backend)
    #   orchestrator = Orchestrator(llm=llm, tools=tools)
    #   result = orchestrator.run(task, context={"run_card": run_card.dict()})
    #   print("\n=== Agent result ===")
    #   print(result)

    print("\n=== Available DQM tools ===")
    for t in tools:
        print(f"  {t.name:40s}  {t.description[:60]}...")

    print("\n=== Agent task ===")
    print(task.strip())
    print("\n(Orchestrator integration pending — wire up via Orchestral AI engine)")


if __name__ == "__main__":
    main()
