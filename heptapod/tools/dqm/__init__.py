"""
heptapod.tools.dqm
==================
ML4DQM tools — a HEPTAPOD extension for CMS Data Quality Monitoring.

Organises tools into four layers mirroring the DQM pipeline:

    data/        – fetch detector histograms and run metadata
    models/      – anomaly detection and run-quality ML models
    training/    – tooling to train / retrain models on labelled runs
    deployment/  – real-time monitoring hooks and shifter alerts

All tools follow the standard HEPTAPOD interface:
  • RuntimeFields  (pydantic BaseModel) – per-invocation inputs
  • StateFields    (pydantic BaseModel) – mutable agent state
  • execute()      – main entry point consumed by the orchestrator
"""

from heptapod.tools.dqm.data.run_registry import RunRegistryTool
from heptapod.tools.dqm.data.dqm_gui import DQMGuiFetchTool
from heptapod.tools.dqm.data.root_reader import RootHistogramTool
from heptapod.tools.dqm.models.autoencoder import AutoencoderAnomalyTool
from heptapod.tools.dqm.models.reference_compare import ReferenceCompareTool
from heptapod.tools.dqm.models.classifier import RunQualityClassifierTool
from heptapod.tools.dqm.training.trainer import ModelTrainerTool
from heptapod.tools.dqm.training.data_prep import HistogramFeatureTool
from heptapod.tools.dqm.deployment.monitor import OnlineMonitorTool
from heptapod.tools.dqm.deployment.alert import ShifterAlertTool
from heptapod.tools.dqm.deployment.run_card_dqm import DQMRunCard

__all__ = [
    # Data access
    "RunRegistryTool",
    "DQMGuiFetchTool",
    "RootHistogramTool",
    # Models
    "AutoencoderAnomalyTool",
    "ReferenceCompareTool",
    "RunQualityClassifierTool",
    # Training
    "ModelTrainerTool",
    "HistogramFeatureTool",
    # Deployment
    "OnlineMonitorTool",
    "ShifterAlertTool",
    # Run-card schema
    "DQMRunCard",
]
