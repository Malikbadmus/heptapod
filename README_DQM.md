# ML4DQM — HEPTAPOD Extension for CMS Data Quality Monitoring

This directory extends the HEPTAPOD framework with a specialised toolset
for **CMS Data Quality Monitoring (DQM)**. It is a GSoC 2026 contribution
to the [ML4SCI ML4DQM project](https://ml4sci.org/gsoc/projects/2026/project_ML4DQM.html).

---

## Motivation

CMS produces ~1 GB/s of detector data during LHC collisions. Human shifters
monitor dozens of histograms per sub-detector, per lumisection (~23 s), to
catch detector problems before they corrupt large fractions of the dataset.
This is cognitively demanding, error-prone, and does not scale with increasing
LHC luminosity.

HEPTAPOD provides an orchestration framework that wraps domain-specific HEP
tools as typed, schema-validated Python classes and presents them to an LLM
orchestrator. This extension applies that framework to DQM — giving the LLM
agent access to a curated set of CMS-specific tools while preserving full
human oversight over certification decisions.

---

## Folder structure

```
heptapod/tools/dqm/
│
├── __init__.py                   # top-level registry of all DQM tools
│
├── data/                         # layer 1 — data access
│   ├── run_registry.py           #   CMS Run Registry REST client
│   ├── dqm_gui.py                #   DQM GUI JSON fairy client
│   └── root_reader.py            #   uproot-based .root file reader
│
├── models/                       # layer 2 — anomaly detection / classification
│   ├── autoencoder.py            #   unsupervised autoencoder scorer
│   ├── reference_compare.py      #   KS / chi² / BDM reference test
│   └── classifier.py             #   run-level GOOD/UNCERTAIN/BAD classifier
│
├── training/                     # layer 3 — model (re)training
│   ├── data_prep.py              #   histogram → feature vector extraction
│   └── trainer.py                #   autoencoder & classifier training loop
│
└── deployment/                   # layer 4 — real-time integration
    ├── run_card_dqm.py           #   DQMRunCard schema (orchestration boundary)
    ├── monitor.py                #   per-lumisection online monitoring loop
    └── alert.py                  #   structured shifter alert dispatcher

examples/
└── dqm_demo.py                   # end-to-end agent demo

prompts/
└── dqm_system_prompt.txt         # LLM system prompt for the DQM agent
```

The layer structure mirrors the actual CMS DQM pipeline:

```
Run Registry ──► Histogram Fetch ──► Feature Extraction ──► Anomaly Detection
                                                                     │
                                           Shifter Alert ◄── Run Classification
```

---

## Design decisions

### 1. DQMRunCard as the orchestration boundary

Every DQM agent invocation is parameterised by a `DQMRunCard` — a Pydantic
model that captures the run number, subsystem, reference run, model
checkpoint, alert thresholds, and operating mode. This directly inherits
HEPTAPOD's run-card philosophy: every decision is auditable, reproducible,
and version-controllable.

```python
from heptapod.tools.dqm.deployment.run_card_dqm import DQMRunCard, DQMMode, Subsystem

card = DQMRunCard(
    run_number=380238,
    subsystem=Subsystem.ECAL,
    reference_run=379100,
    mode=DQMMode.OFFLINE,
    anomaly_threshold=0.95,
    dry_run=False,
)
```

### 2. Online/offline split with the same tool interface

The same tools work in both modes — the `DQMMode` field on the run card
controls behaviour. Online mode enforces a strict per-lumisection latency
budget (~5 s total); the `OnlineMonitorTool` wraps this loop and tracks
consecutive-bad-LS counters. Offline mode uses the same tools but adds
the full training pipeline.

### 3. Two complementary anomaly detectors

| Tool | Method | Catches |
|------|--------|---------|
| `AutoencoderAnomalyTool` | Reconstruction error (MSE) | Global shape anomalies; distributions that look nothing like good runs |
| `ReferenceCompareTool` | KS / chi² / BDM test | Local deviations from a specific reference run; calibration shifts |

Using both in parallel provides complementary coverage: the autoencoder
catches novel failure modes with no reference run available; the reference
test catches subtle shifts a trained AE might not flag.

### 4. Human oversight preserved throughout

The agent **never** autonomously:
- Certifies a run in the Run Registry
- Stops data-taking
- Retrains a model on uncertified data

`ShifterAlertTool` always dispatches to a human first. The `dry_run` flag
on `DQMRunCard` allows the full pipeline to execute in test mode without
emitting external side effects.

### 5. Schema-validated tools (HEPTAPOD convention)

Every tool exposes `RuntimeFields` and `StateFields` as Pydantic `BaseModel`
subclasses with explicit type annotations and docstrings. These are
automatically translated into JSON schemas by the HEPTAPOD orchestrator,
ensuring the LLM always receives a machine-readable description of available
inputs and their constraints.

---

## Usage

### Offline run assessment

```python
from heptapod.tools.dqm import (
    RunRegistryTool, DQMGuiFetchTool, AutoencoderAnomalyTool,
    ReferenceCompareTool, HistogramFeatureTool,
    RunQualityClassifierTool, ShifterAlertTool,
)
from heptapod.tools.dqm.deployment.run_card_dqm import DQMRunCard, Subsystem, DQMMode

# Build the run card
card = DQMRunCard(
    run_number=380238,
    subsystem=Subsystem.ECAL,
    reference_run=379100,
    mode=DQMMode.OFFLINE,
)

# Step 1: Query run metadata
registry = RunRegistryTool()
meta = registry.execute(
    registry.RuntimeFields(run_number=card.run_number),
    registry.StateFields(),
)

# Step 2: Fetch histograms
fetcher = DQMGuiFetchTool()
histograms = fetcher.execute(
    fetcher.RuntimeFields(
        run_number=card.run_number,
        dataset="/PromptReco/Run2024G/DQM",
        histogram_paths=["EcalBarrel/EBOccupancyTask/EBOT digi occupancy"],
    ),
    fetcher.StateFields(),
)

# Step 3–6: score → feature → classify → alert (see dqm_demo.py for full pipeline)
```

### Demo

```bash
python examples/dqm_demo.py --run 380238 --subsystem ecal --mode offline --dry-run
```

### Online monitoring loop

```python
from heptapod.tools.dqm.deployment.monitor import OnlineMonitorTool

monitor = OnlineMonitorTool()
state = monitor.StateFields(current_run=380238)

for ls in range(1, 200):
    result = monitor.execute(
        monitor.RuntimeFields(run_number=380238, lumisection=ls, subsystem="ecal"),
        state,
    )
    if result["recommendation"] == "STOP":
        print(f"Agent recommends stopping run at LS {ls}")
        break
```

---

## Dependencies

Core (required):
```
pydantic>=2.0
numpy
scipy
```

Data access:
```
uproot          # ROOT file reading without ROOT installation
requests        # Run Registry and DQM GUI REST calls
```

ML inference:
```
torch           # autoencoder inference
scikit-learn    # classifier, PCA feature extraction
```

Training (optional):
```
xgboost         # alternative to sklearn GradientBoostingClassifier
joblib          # model serialisation
```

Install:
```bash
pip install pydantic numpy scipy uproot requests torch scikit-learn
```

---

## Relationship to HEPTAPOD

This extension follows all HEPTAPOD tool conventions:

| Convention | Implementation |
|------------|---------------|
| `RuntimeFields` Pydantic model | Every tool defines `RuntimeFields` |
| `StateFields` Pydantic model | Every tool defines `StateFields` |
| `execute(runtime, state) → dict` | All tools implement `execute()` |
| Run-card-driven orchestration | `DQMRunCard` is the canonical config object |
| JSON-schema-compatible types | All fields use Pydantic with explicit annotations |
| Human-in-the-loop | `ShifterAlertTool` required for any certification action |

The DQM system prompt (`prompts/dqm_system_prompt.txt`) conditions the
orchestrating LLM on CMS domain knowledge, tool usage rules, and the
human oversight constraints.

---

## References

- Pol et al., "Anomaly detection with conditional variational autoencoders", [arXiv:1911.10461](https://arxiv.org/abs/1911.10461)
- HEPTAPOD paper: [arXiv:2512.15867](https://arxiv.org/abs/2512.15867)
- CMS DQM documentation: https://cms-dqm.web.cern.ch
- CMS Run Registry: https://cms-run-registry.web.cern.ch
- uproot: https://uproot.readthedocs.io
