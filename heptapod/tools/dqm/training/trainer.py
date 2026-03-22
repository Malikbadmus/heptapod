"""
heptapod.tools.dqm.training.trainer
=====================================
Tool for training and retraining ML4DQM models.

The trainer orchestrates the full training loop for both:
  * Autoencoder (unsupervised, trained on GOOD-certified runs only)
  * Run quality classifier (supervised, requires certified labels)

Models are saved as checkpoints that can be passed to inference tools
via DQMRunCard.model_checkpoint.

Training data format
--------------------
A labelled training corpus is a list of records:
  { "run_number": int, "label": "GOOD"|"BAD", "features": list[float] }

For autoencoder training only GOOD-labelled records are used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelTrainerRuntimeFields(BaseModel):
    """Inputs consumed per invocation."""

    training_records: list[dict] = Field(
        ...,
        min_length=1,
        description=(
            "List of {run_number, label, features} dicts. "
            "Label must be 'GOOD', 'BAD', or 'UNCERTAIN'."
        ),
    )
    model_type: str = Field(
        default="autoencoder",
        description="Model to train: 'autoencoder' or 'classifier'.",
    )
    output_checkpoint_path: str = Field(
        ...,
        description="File path where the trained model will be saved.",
    )
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    latent_dim: int = Field(
        default=16,
        ge=2,
        description="Autoencoder bottleneck dimension (ignored for classifier).",
    )
    validation_split: float = Field(default=0.2, ge=0.0, le=0.5)


class ModelTrainerStateFields(BaseModel):
    """Agent state updated after each invocation."""

    last_train_loss: Optional[float] = None
    last_val_loss: Optional[float] = None
    saved_checkpoint: Optional[str] = None


class ModelTrainerTool:
    """Train or retrain an ML4DQM model on a labelled run corpus.

    Supports continuous re-training as new certified runs accumulate —
    a critical capability for DQM because detector conditions evolve
    throughout a data-taking period.

    The agent invokes this tool when:
      * The UNCERTAIN rate from RunQualityClassifierTool is rising
        (distribution shift detected).
      * A human shifter manually certifies a batch of new runs.
      * The autoencoder reconstruction error distribution drifts.

    Usage
    -----
    >>> tool = ModelTrainerTool()
    >>> result = tool.execute(
    ...     ModelTrainerRuntimeFields(
    ...         training_records=[{"run_number": 380200, "label": "GOOD",
    ...                            "features": [...]}],
    ...         model_type="autoencoder",
    ...         output_checkpoint_path="/models/ae_v2.pt",
    ...     ),
    ...     ModelTrainerStateFields(),
    ... )
    """

    name: str = "model_trainer"
    description: str = (
        "Train or retrain an ML4DQM model (autoencoder or classifier) on a "
        "corpus of labelled CMS DQM runs and save the checkpoint."
    )

    RuntimeFields = ModelTrainerRuntimeFields
    StateFields = ModelTrainerStateFields

    def execute(
        self,
        runtime: ModelTrainerRuntimeFields,
        state: ModelTrainerStateFields,
    ) -> dict:
        """Run the training loop and save the model checkpoint.

        Parameters
        ----------
        runtime:
            Training records, model type, hyperparameters, output path.
        state:
            Updated with final train/val loss and checkpoint path.

        Returns
        -------
        dict
            Keys: train_loss, val_loss, n_train, n_val, checkpoint_path.

        Raises
        ------
        ValueError
            If model_type is not recognised or training data is empty.
        RuntimeError
            If training diverges (loss becomes NaN).
        """
        import numpy as np

        records = runtime.training_records
        n_total = len(records)
        n_val = max(1, int(n_total * runtime.validation_split))
        n_train = n_total - n_val

        if runtime.model_type == "autoencoder":
            good_records = [r for r in records if r.get("label") == "GOOD"]
            if not good_records:
                raise ValueError("Autoencoder requires at least one GOOD-labelled run.")
            # TODO: implement PyTorch training loop
            # from heptapod.tools.dqm.models._architectures import DQMAutoencoder
            # model = DQMAutoencoder(input_dim=len(good_records[0]["features"]),
            #                        latent_dim=runtime.latent_dim)
            # train_ae(model, good_records, runtime)
            train_loss, val_loss = 0.042, 0.051  # stub

        elif runtime.model_type == "classifier":
            # TODO: implement sklearn/XGBoost training
            # from sklearn.ensemble import GradientBoostingClassifier
            # X = np.array([r["features"] for r in records])
            # y = np.array([label_to_int(r["label"]) for r in records])
            # model = GradientBoostingClassifier(...)
            # model.fit(X[:n_train], y[:n_train])
            train_loss, val_loss = 0.08, 0.11  # stub

        else:
            raise ValueError(f"Unknown model_type: {runtime.model_type!r}")

        # TODO: save checkpoint, e.g. torch.save(model.state_dict(), path)
        Path(runtime.output_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(runtime.output_checkpoint_path).write_text("# stub checkpoint\n")

        state.last_train_loss = train_loss
        state.last_val_loss = val_loss
        state.saved_checkpoint = runtime.output_checkpoint_path

        return {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "n_train": n_train,
            "n_val": n_val,
            "checkpoint_path": runtime.output_checkpoint_path,
        }
