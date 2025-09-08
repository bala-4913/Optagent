%%writefile mfb_opt/model_interface.py
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
    MFB ModelInterface with support for two schemas:

    1) Autoregressive (AR) schema — detected when feature_dim >= 7 and (feature_dim - 7) is even.
       Features per step:
         [on, off, vel, cycle_time, duty_cycle]              (5)
         [current_time, time_in_cycle]                       (2)
         seg_sequence (length = seq_len)                     (seq_len)
         deltas       (length = seq_len)                     (seq_len)
       Total = 7 + 2*seq_len = feature_dim (e.g., 27 with seq_len=10)

    2) Static per-time-step schema (fallback):
         [on, off, vel, cycle_time] (+ optional phase)      (4 or 5)
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
        # Ensure path strings
        self.model_path = str(self.model_path)
        self.scaler_feat_path = str(self.scaler_feat_path)
        self.scaler_tgt_path = str(self.scaler_tgt_path)

        self._model = None
        self._scaler_feat = _IdentityScaler()
        self._scaler_tgt = _IdentityScaler()
        self._feature_dim: Optional[int] = None
        self._is_ar_schema: bool = False
        self._seq_len: Optional[int] = None
        self._dt_for_ar: Optional[float] = None  # If we want to override dt for AR
        self._cache = _LRUCache(self.cache_max_size)
        self._rng = np.random.default_rng(42)

        self._load_artifacts()
        self._infer_schema()

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

        If the model scaler indicates AR schema (e.g., feature_dim=27), we roll out
        autoregressively using seq_len inferred from the scaler.
        """
        on, off, vel, period = self.validate_parameters((on_time, off_time, velocity, None))
        dt = self._dt_for_ar if (self._is_ar_schema and self._dt_for_ar is not None) else self.default_dt

        key = self._build_cache_key(on, off, vel, period, n_cycles, dt, self._seq_len or -1, self._is_ar_schema)
        cached = self._cache.get(key)
        if cached is not None:
            t, y = cached
            return {"t": t.copy(), "seg_index": y.copy()}

        if self._is_ar_schema:
            t, y = self._predict_autoregressive(on, off, vel, period, n_cycles, dt)
        else:
            t = self._build_time_axis(period, n_cycles, dt)
            X = self._build_feature_matrix_static(on, off, vel, period, t)
            y = self._predict_matrix(X, fallback_ok=True)

        self._cache.put(key, (t, y))
        return {"t": t, "seg_index": y}

    def batch_predict(self, parameter_array: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """
        Batch prediction for an array of parameters.
        parameter_array: shape (N, 3) or (N, 4): [on, off, vel, (optional) cycle_time]
        Returns list of dicts {'t', 'seg_index'} for each row.

        Note: For AR schema, rollout is inherently sequential per sample,
        so we loop per row (still cached).
        """
        from numpy import ndarray
        if not isinstance(parameter_array, ndarray) or parameter_array.ndim != 2:
            raise ParameterValidationError("parameter_array must be a 2D numpy array with shape (N, 3) or (N, 4).")
        if parameter_array.shape[1] not in (3, 4):
            raise ParameterValidationError("parameter_array must have 3 or 4 columns: [on, off, vel, (cycle_time)].")

        N = parameter_array.shape[0]
        outputs: List[Optional[Dict[str, np.ndarray]]] = [None] * N  # type: ignore
        dt = self._dt_for_ar if (self._is_ar_schema and self._dt_for_ar is not None) else self.default_dt

        for i in range(N):
            row = parameter_array[i]
            on, off, vel, period = self.validate_parameters((row[0], row[1], row[2], row[3] if parameter_array.shape[1] == 4 else None))
            key = self._build_cache_key(on, off, vel, period, self.default_cycles, dt, self._seq_len or -1, self._is_ar_schema)
            cached = self._cache.get(key)
            if cached is not None:
                t, y = cached
                outputs[i] = {"t": t.copy(), "seg_index": y.copy()}
                continue

            if self._is_ar_schema:
                t, y = self._predict_autoregressive(on, off, vel, period, self.default_cycles, dt)
            else:
                t = self._build_time_axis(period, self.default_cycles, dt)
                X = self._build_feature_matrix_static(on, off, vel, period, t)
                y = self._predict_matrix(X, fallback_ok=True)

            self._cache.put(key, (t, y))
            outputs[i] = {"t": t, "seg_index": y}

        return outputs  # type: ignore

    def validate_parameters(
        self, params: Union[Tuple[float, float, float, Optional[float]], List[float]]
    ) -> Tuple[float, float, float, float]:
        """
        Validate and normalize params: (on_time, off_time, velocity, cycle_time [optional]).
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
            "schema": "autoregressive" if self._is_ar_schema else "static",
            "seq_len": self._seq_len,
            "default_dt": self.default_dt,
            "dt_for_ar": self._dt_for_ar,
            "default_cycles": self.default_cycles,
            "bounds": {"on_time": self.on_bounds, "off_time": self.off_bounds, "velocity": self.vel_bounds},
        }

    # ------------------ Internal helpers ------------------
    def _load_artifacts(self):
        # Keras load (optional)
        try:
            from tensorflow.keras.models import load_model  # noqa: F401
        except Exception as e:
            logger.warning("TensorFlow/Keras not available: %s. Using fallback predictor.", e)
            self._model = None
        else:
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
            if os.path.exists(self.scaler_feat_path):
                self._scaler_feat = _load_pkl(self.scaler_feat_path)
            else:
                logger.warning("Feature scaler missing at %s. Using identity.", self.scaler_feat_path)
                self._scaler_feat = _IdentityScaler()
        except Exception as e:
            logger.warning("Failed to load feature scaler: %s. Using identity.", e)
            self._scaler_feat = _IdentityScaler()

        try:
            if os.path.exists(self.scaler_tgt_path):
                self._scaler_tgt = _load_pkl(self.scaler_tgt_path)
            else:
                logger.warning("Target scaler missing at %s. Using identity.", self.scaler_tgt_path)
                self._scaler_tgt = _IdentityScaler()
        except Exception as e:
            logger.warning("Failed to load target scaler: %s. Using identity.", e)
            self._scaler_tgt = _IdentityScaler()

        # Infer feature dims if possible
        self._feature_dim = None
        for attr in ("n_features_in_", "mean_", "scale_", "data_min_", "data_max_"):
            if hasattr(self._scaler_feat, attr):
                val = getattr(self._scaler_feat, attr)
                if isinstance(val, np.ndarray):
                    self._feature_dim = int(val.shape[0])
                    break

    def _infer_schema(self):
        """
        Decide whether the model is AR (27, etc.) or static (4/5).
        Also infer seq_len if AR.
        """
        fd = self._feature_dim
        if fd is None:
            self._is_ar_schema = False
            self._seq_len = None
            return

        # AR: fd = 7 + 2*seq_len
        if fd >= 7 and (fd - 7) % 2 == 0:
            seq_len = (fd - 7) // 2
            if seq_len >= 1:
                self._is_ar_schema = True
                self._seq_len = int(seq_len)
                # Most AR trainings used dt=0.1 in your examples; set a sensible default if DEFAULT_DT differs
                if abs(self.default_dt - 0.1) > 1e-9:
                    self._dt_for_ar = 0.1
                logger.info("Detected AR schema: feature_dim=%d, seq_len=%d, dt_for_ar=%s",
                            fd, self._seq_len, str(self._dt_for_ar))
                return

        # Otherwise static schema (4/5 features per time)
        self._is_ar_schema = False
        self._seq_len = None

    @staticmethod
    def _build_time_axis(period: float, n_cycles: int, dt: float) -> np.ndarray:
        total = n_cycles * period
        n_steps = int(np.floor(total / dt)) + 1
        return np.linspace(0.0, total, n_steps)

    # ----- STATIC schema -----
    def _build_feature_matrix_static(self, on: float, off: float, vel: float, period: float, t: np.ndarray) -> np.ndarray:
        n = t.shape[0]
        base = np.column_stack([
            np.full(n, on, dtype=float),
            np.full(n, off, dtype=float),
            np.full(n, vel, dtype=float),
            np.full(n, period, dtype=float),
        ])
        # If scaler expects >=5, provide phase
        if self._feature_dim is None or self._feature_dim >= 5:
            phase = (np.mod(t, period) / max(period, 1e-12)).reshape(-1, 1)
            return np.hstack([base, phase])
        return base

    # ----- AR schema -----
    def _predict_autoregressive(self, on: float, off: float, vel: float, period: float, n_cycles: int, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Roll-out using the AR feature schema (5 static + 2 time + seq_len + deltas).
        """
        assert self._is_ar_schema and self._seq_len is not None and self._feature_dim is not None

        total_time = n_cycles * period
        t_all = np.arange(0.0, total_time + dt, dt, dtype=float)
        # Seed initial sequence (simple deterministic seed near 0.5 with slight ramp/noise)
        init_seq = self._create_initial_seq(on, off, vel, dt, self._seq_len, period)

        # Initialize trajectory: include initial seq
        traj = list(init_seq)

        # Autoregressive loop
        for i, t_now in enumerate(t_all[self._seq_len:]):
            t_window = t_all[i:i + self._seq_len]
            x_row = self._build_ar_feature_row(init_seq=np.array(traj[-self._seq_len:], dtype=float),
                                               time_seq=t_window, on=on, off=off, vel=vel, period=period)
            pred = self._predict_next(x_row, on, off, vel, t_now, period)
            traj.append(pred)

        return t_all, np.asarray(traj, dtype=float)

    def _create_initial_seq(self, on: float, off: float, vel: float, dt: float, seq_len: int, period: float,
                            initial_segregation: float = 0.5) -> np.ndarray:
        seq = []
        for i in range(seq_len):
            t = i * dt
            t_in = t % period
            if t_in <= on:
                base = initial_segregation + 0.3 * (i / max(seq_len, 1))
            else:
                base = initial_segregation - 0.2 * (i / max(seq_len, 1))
            noise = float(self._rng.normal(0.0, 0.02))
            seq.append(np.clip(base + noise, 0.0, 1.0))
        return np.array(seq, dtype=float)

    def _build_ar_feature_row(self, init_seq: np.ndarray, time_seq: np.ndarray, on: float, off: float, vel: float, period: float) -> np.ndarray:
        duty = on / max(period, 1e-12)
        current_time = float(time_seq[-1])
        time_in_cycle = float(current_time % period if period > 0 else 0.0)
        deltas = np.diff(time_seq, prepend=time_seq[0]).astype(np.float32)

        static = [float(on), float(off), float(vel), float(period), float(np.clip(duty, 0.0, 1.0))]
        feats = static + [current_time, time_in_cycle] + init_seq.astype(np.float32).tolist() + deltas.tolist()

        x = np.asarray(feats, dtype=np.float32).reshape(1, -1)

        # Sanity: match scaler's expected feature count
        if self._feature_dim is not None and x.shape[1] != self._feature_dim:
            raise RuntimeError(f"AR feature length {x.shape[1]} does not match expected {self._feature_dim}. "
                               f"Check seq_len/time_step or training schema.")
        return x

    def _predict_next(self, x_row: np.ndarray, on: float, off: float, vel: float, t_now: float, period: float) -> float:
        # Feature scaling
        x_scaled = x_row
        if self._scaler_feat is not None:
            try:
                x_scaled = self._scaler_feat.transform(x_row)
            except Exception as e:
                logger.error("Feature scaling failed (AR): %s. Using unscaled.", e)

        # Model inference or fallback
        if self._model is None:
            pred = self._fallback_step(on, off, vel, t_now, period)
        else:
            try:
                y_scaled = self._model.predict(x_scaled, verbose=0)
                y_scaled = np.asarray(y_scaled)
                if y_scaled.ndim == 2 and y_scaled.shape[1] == 1:
                    y_scaled = y_scaled.ravel()
                elif y_scaled.ndim > 1:
                    y_scaled = y_scaled[:, 0].ravel()
                pred = y_scaled.astype(float)
                if self._scaler_tgt is not None:
                    try:
                        pred2 = self._scaler_tgt.inverse_transform(pred.reshape(-1, 1))
                        pred = np.ravel(pred2)
                    except Exception as e:
                        logger.warning("Target inverse_transform failed: %s. Using raw model outputs.", e)
                pred = float(np.clip(pred[0], 0.0, 1.0))
            except Exception as e:
                logger.error("Model prediction failed (AR): %s. Using fallback.", e)
                pred = self._fallback_step(on, off, vel, t_now, period)

        return pred

    # ------------------ Generic prediction core for STATIC schema ------------------
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

    def _fallback_step(self, on: float, off: float, vel: float, t_now: float, period: float) -> float:
        t_in = t_now % period if period > 0 else 0.0
        duty = np.clip(on / max(period, 1e-12), 0.0, 1.0)
        vel_n = np.clip((vel - 0.7) / 0.3, 0.0, 1.0)
        base = 0.35 + 0.25 * (1.0 - vel_n)
        amp = 0.12 * (1.0 - 0.4 * (duty - 0.5))
        phase = (t_in / max(period, 1e-12)) if period > 0 else 0.0
        wave = np.sin(2 * np.pi * phase) - 0.25 * np.sin(4 * np.pi * phase)
        trend = -0.05 * vel_n
        return float(np.clip(base + amp * wave + trend, 0.0, 1.0))

    # ------------------ Cache key ------------------
    def _build_cache_key(self, on: float, off: float, vel: float, period: float, n_cycles: int, dt: float, seq_len: int, is_ar: bool):
        r = self.cache_key_round
        return (round(on, r), round(off, r), round(vel, r), round(period, r), int(n_cycles), round(dt, r), int(seq_len), int(is_ar))
