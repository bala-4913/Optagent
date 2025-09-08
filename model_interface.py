# mfb_opt/model_interface.py
from __future__ import annotations

import os
import pickle
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np

from .config import MODEL_PATH, SCALER_FEAT_PATH, SCALER_TGT_PATH, DEFAULT_DT, DEFAULT_CYCLES, BOUNDS
from .utils.logging_utils import get_logger

logger = get_logger("MFB.ModelInterface")


class ModelLoadError(RuntimeError):
    pass


class ParameterValidationError(ValueError):
    pass


class _IdentityScaler:
    def transform(self, X): return X
    def inverse_transform(self, y): return y
    @property
    def n_features_in_(self): return None


class _LRUCache:
    def __init__(self, max_size: int = 512):
        from collections import OrderedDict
        self._store = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size

    def get(self, key):
        with self._lock:
            if key not in self._store:
                return None
            v = self._store.pop(key)
            self._store[key] = v
            return v

    def put(self, key, value):
        with self._lock:
            if key in self._store:
                self._store.pop(key)
            self._store[key] = value
            if len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self):
        with self._lock:
            self._store.clear()


@dataclass
class ModelInterface:
    """
    Loads a pre-trained FFNN and scalers to predict segregation index time series.

    INPUT: [on_time, off_time, velocity, cycle_time]
    OUTPUT: dict {'t': np.ndarray, 'seg_index': np.ndarray} for n_cycles (default 3)

    Training context (documentation):
      - Trained on 139 CFD-DEM simulations, validated on 30
      - Each simulation spans 3 cycles
    """
    model_path: str = MODEL_PATH
    scaler_feat_path: str = SCALER_FEAT_PATH
    scaler_tgt_path: str = SCALER_TGT_PATH
    default_dt: float = DEFAULT_DT
    default_cycles: int = DEFAULT_CYCLES
    cache_max_size: int = 512
    cache_key_round: int = 3

    on_bounds: Tuple[float, float] = BOUNDS["on_time"]
    off_bounds: Tuple[float, float] = BOUNDS["off_time"]
    vel_bounds: Tuple[float, float] = BOUNDS["velocity"]

    def __post_init__(self):
        self._model = None
        self._scaler_feat = _IdentityScaler()
        self._scaler_tgt = _IdentityScaler()
        self._feature_dim: Optional[int] = None
        self._cache = _LRUCache(self.cache_max_size)
        self._rng = np.random.default_rng(42)

        self._load_artifacts()

    # ------------------ Public API ------------------
    def predict_segregation_trajectory(
        self,
        on_time: float,
        off_time: float,
        velocity: float,
        n_cycles: int = 3
    ) -> Dict[str, np.ndarray]:
        """
        Predict segregation trajectory for n_cycles (default=3).
        cycle_time is inferred as on_time + off_time.
        """
        on, off, vel, period = self.validate_parameters((on_time, off_time, velocity, None))
        key = self._build_cache_key(on, off, vel, period, n_cycles, self.default_dt)
        cached = self._cache.get(key)
        if cached is not None:
            t, y = cached
            return {"t": t.copy(), "seg_index": y.copy()}

        t = self._build_time_axis(period, n_cycles, self.default_dt)
        X = self._build_feature_matrix(on, off, vel, period, t)
        y = self._predict_matrix(X, fallback_ok=True)

        self._cache.put(key, (t, y))
        return {"t": t, "seg_index": y}

    def batch_predict(self, parameter_array: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Batch prediction for an array of parameters.
        parameter_array: shape (N, 3) or (N, 4): [on, off, vel, (optional) cycle_time]
        Returns list of dicts {'t', 'seg_index'} for each row.
        """
        if not isinstance(parameter_array, np.ndarray) or parameter_array.ndim != 2:
            raise ParameterValidationError("parameter_array must be a 2D numpy array with shape (N, 3) or (N, 4).")
        if parameter_array.shape[1] not in (3, 4):
            raise ParameterValidationError("parameter_array must have 3 or 4 columns: [on, off, vel, (cycle_time)].")

        N = parameter_array.shape[0]
        outputs: List[Optional[Dict[str, np.ndarray]]] = [None] * N  # type: ignore
        compute_specs = []
        cache_keys = []

        for i in range(N):
            row = parameter_array[i]
            on, off, vel, period = self.validate_parameters((row[0], row[1], row[2], row[3] if parameter_array.shape[1] == 4 else None))
            key = self._build_cache_key(on, off, vel, period, self.default_cycles, self.default_dt)
            cache_keys.append(key)
            cached = self._cache.get(key)
            if cached is not None:
                t, y = cached
                outputs[i] = {"t": t.copy(), "seg_index": y.copy()}
            else:
                t = self._build_time_axis(period, self.default_cycles, self.default_dt)
                compute_specs.append((i, on, off, vel, period, t))

        if not compute_specs:
            return outputs  # type: ignore

        # Build stacked feature matrix
        X_blocks = []
        slices = []
        start = 0
        for (i, on, off, vel, period, t) in compute_specs:
            X_i = self._build_feature_matrix(on, off, vel, period, t)
            X_blocks.append(X_i)
            end = start + X_i.shape[0]
            slices.append((start, end, i, t))
            start = end

        X_all = np.vstack(X_blocks)
        y_all = self._predict_matrix(X_all, fallback_ok=True)

        for (start, end, i, t) in slices:
            y_i = y_all[start:end]
            outputs[i] = {"t": t, "seg_index": y_i}
            self._cache.put(cache_keys[i], (t, y_i))

        return outputs  # type: ignore

    def validate_parameters(
        self, params: Union[Tuple[float, float, float, Optional[float]], List[float]]
    ) -> Tuple[float, float, float, float]:
        """
        Validate and normalize parms: (on_time, off_time, velocity, cycle_time [optional]).
        Returns (on, off, vel, period).
        """
        if params is None or len(params) < 3:
            raise ParameterValidationError("params must include at least (on_time, off_time, velocity).")
        on, off, vel = float(params[0]), float(params[1]), float(params[2])
        cyc = float(params[3]) if len(params) >= 4 and params[3] is not None else None

        if not (self.on_bounds[0] <= on <= self.on_bounds[1]):
            raise ParameterValidationError(f"on_time out of bounds [{self.on_bounds[0]}, {self.on_bounds[1]}]: {on}")
        if not (self.off_bounds[0] <= off <= self.off_bounds[1]):
            raise ParameterValidationError(f"off_time out of bounds [{self.off_bounds[0]}, {self.off_bounds[1]}]: {off}")
        if not (self.vel_bounds[0] <= vel <= self.vel_bounds[1]):
            raise ParameterValidationError(f"velocity out of bounds [{self.vel_bounds[0]}, {self.vel_bounds[1]}]: {vel}")

        period = on + off if cyc is None else float(cyc)
        if period <= 0:
            raise ParameterValidationError("cycle_time must be positive (on+off or provided).")
        return on, off, vel, period

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_loaded": self._model is not None,
            "model_path": self.model_path,
            "feature_scaler_loaded": not isinstance(self._scaler_feat, _IdentityScaler),
            "target_scaler_loaded": not isinstance(self._scaler_tgt, _IdentityScaler),
            "feature_dim": self._feature_dim,
            "default_dt": self.default_dt,
            "default_cycles": self.default_cycles,
            "bounds": {"on_time": self.on_bounds, "off_time": self.off_bounds, "velocity": self.vel_bounds},
        }

    # ------------------ Internal helpers ------------------
    def _load_artifacts(self):
        # Lazy import TF to avoid hard dependency during read-only runs
        try:
            from tensorflow.keras.models import load_model  # noqa: F401
        except Exception as e:
            logger.warning("TensorFlow/Keras not available: %s. Using fallback predictor.", e)
            self._model = None
        else:
            # Load model file
            if os.path.exists(self.model_path):
                try:
                    from tensorflow.keras.models import load_model
                    self._model = load_model(self.model_path)
                    logger.info("Loaded FFNN model: %s", self.model_path)
                except Exception as e:
                    logger.error("Failed to load Keras model (%s). Using fallback predictor.", e)
                    self._model = None
            else:
                logger.warning("Model file not found at %s. Using fallback predictor.", self.model_path)
                self._model = None

        # Load scalers
        def _load_pkl(path: str):
            with open(path, "rb") as f:
                return pickle.load(f)

        try:
            self._scaler_feat = _load_pkl(self.scaler_feat_path) if os.path.exists(self.scaler_feat_path) else _IdentityScaler()
            if isinstance(self._scaler_feat, _IdentityScaler):
                logger.warning("Feature scaler missing or not loaded. Using identity transform.")
        except Exception as e:
            logger.warning("Failed to load feature scaler: %s. Using identity transform.", e)
            self._scaler_feat = _IdentityScaler()

        try:
            self._scaler_tgt = _load_pkl(self.scaler_tgt_path) if os.path.exists(self.scaler_tgt_path) else _IdentityScaler()
            if isinstance(self._scaler_tgt, _IdentityScaler):
                logger.warning("Target scaler missing or not loaded. Using identity transform.")
        except Exception as e:
            logger.warning("Failed to load target scaler: %s. Using identity transform.", e)
            self._scaler_tgt = _IdentityScaler()

        # Infer feature dims if possible
        self._feature_dim = None
        for attr in ("n_features_in_", "mean_", "scale_", "data_min_", "data_max_"):
            if hasattr(self._scaler_feat, attr):
                val = getattr(self._scaler_feat, attr)
                if isinstance(val, np.ndarray):
                    self._feature_dim = int(val.shape[0])
                    break

    @property
    def model_path(self): return self._model_path if hasattr(self, "_model_path") else MODEL_PATH
    @model_path.setter
    def model_path(self, p): self._model_path = p

    @property
    def scaler_feat_path(self): return self._scaler_feat_path if hasattr(self, "_scaler_feat_path") else SCALER_FEAT_PATH
    @scaler_feat_path.setter
    def scaler_feat_path(self, p): self._scaler_feat_path = p

    @property
    def scaler_tgt_path(self): return self._scaler_tgt_path if hasattr(self, "_scaler_tgt_path") else SCALER_TGT_PATH
    @scaler_tgt_path.setter
    def scaler_tgt_path(self, p): self._scaler_tgt_path = p

    @staticmethod
    def _build_time_axis(period: float, n_cycles: int, dt: float) -> np.ndarray:
        total = n_cycles * period
        n_steps = int(np.floor(total / dt)) + 1
        return np.linspace(0.0, total, n_steps)

    def _build_feature_matrix(self, on: float, off: float, vel: float, period: float, t: np.ndarray) -> np.ndarray:
        """
        Feature schema supported:
            [on, off, vel, cycle_time, time_in_cycle_norm]  (preferred)
            or if scaler expects 4: [on, off, vel, cycle_time]
        """
        n = t.shape[0]
        base = np.column_stack([
            np.full(n, on, dtype=float),
            np.full(n, off, dtype=float),
            np.full(n, vel, dtype=float),
            np.full(n, period, dtype=float),
        ])
        if self._feature_dim is None or self._feature_dim >= 5:
            phase = (np.mod(t, period) / max(period, 1e-12)).reshape(-1, 1)
            return np.hstack([base, phase])
        return base

    def _predict_matrix(self, X: np.ndarray, fallback_ok: bool = True) -> np.ndarray:
        try:
            Xs = self._scaler_feat.transform(X)
        except Exception as e:
            logger.error("Feature scaling failed: %s", e)
            if not fallback_ok:
                raise RuntimeError(f"Feature scaling failed: {e}")
            Xs = X

        if self._model is None:
            y = self._fallback_predict_from_features(X)
        else:
            try:
                y = self._model.predict(Xs, verbose=0)
            except Exception as e:
                logger.error("Model prediction failed: %s", e)
                if not fallback_ok:
                    raise RuntimeError(f"Model prediction failed: {e}")
                y = self._fallback_predict_from_features(X)

        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        elif y.ndim > 1:
            y = y[:, 0].ravel()

        try:
            y = self._scaler_tgt.inverse_transform(y.reshape(-1, 1)).ravel()
        except Exception:
            pass

        return np.clip(y.astype(float), 0.0, 1.0)

    # Fallback predictor: smooth, physically-plausible waveform
    def _fallback_predict_from_features(self, X: np.ndarray) -> np.ndarray:
        on, off, vel, period = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        phase = X[:, 4] if X.shape[1] >= 5 else (np.arange(X.shape[0]) / max(np.mean(period), 1e-6)) % 1.0
        vel_n = np.clip((vel - 0.7) / 0.3, 0.0, 1.0)
        phi_on = np.clip(on / np.maximum(period, 1e-9), 0.0, 1.0)
        base = 0.35 + 0.25 * (1.0 - vel_n)
        amp = 0.12 * (1.0 - 0.5 * (phi_on - 0.5))
        wave = np.sin(2 * np.pi * phase) - 0.3 * np.sin(4 * np.pi * phase)
        trend = -0.05 * vel_n
        return np.clip(base + amp * wave + trend, 0.0, 1.0)

    def _build_cache_key(self, on: float, off: float, vel: float, period: float, n_cycles: int, dt: float):
        r = self.cache_key_round
        return (round(on, r), round(off, r), round(vel, r), round(period, r), int(n_cycles), round(dt, r))
