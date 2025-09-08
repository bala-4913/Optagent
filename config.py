# mfb_opt/config.py
import os
from dataclasses import dataclass

# Parameter bounds
BOUNDS = {
    "on_time": (2.0, 8.0),
    "off_time": (2.0, 8.0),
    "velocity": (0.7, 1.0),
}

# Model artifacts
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_ffnn_model.h5")
SCALER_FEAT_PATH = os.path.join(MODEL_DIR, "scaler_feat_ffnn.pkl")
SCALER_TGT_PATH = os.path.join(MODEL_DIR, "scaler_tgt_ffnn.pkl")

DEFAULT_DT = 0.05
DEFAULT_CYCLES = 3
RANDOM_SEED = 42


@dataclass(frozen=True)
class FeatureSpec:
    """
    Reference schema for the FFNN input features used here.

    We support:
      - With phase: [on_time, off_time, velocity, cycle_time, time_in_cycle_norm]
      - Without phase: [on_time, off_time, velocity, cycle_time]
    """
    with_phase: bool = True
