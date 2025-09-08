# mfb_opt/objectives.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Any, Tuple

import numpy as np

from .model_interface import ModelInterface, ParameterValidationError
from .utils.logging_utils import get_logger

logger = get_logger("MFB.ObjectiveManager")


class Direction(str, Enum):
    MIN = "min"
    MAX = "max"


@dataclass
class Constraint:
    """
    Generic constraint representation:
     - description: original NL description
     - func: g(x) -> float (<= 0 feasible). If returns vector, engine will handle each entry.
    """
    description: str
    func: Callable[[np.ndarray, Dict[str, np.ndarray]], np.ndarray]  # (x, pred)-> g<=0
    weight: float = 1.0  # penalty weight if used as penalty


@dataclass
class ObjectiveTerm:
    """One metric to minimize or maximize."""
    name: str                # e.g., 'mean_seg', 'peak_seg', 'final_seg', 'on_time', 'off_time', 'cycle_time'
    direction: Direction = Direction.MIN
    weight: float = 1.0      # for weighted sum in single-objective synthesis


@dataclass
class OptimizationConfig:
    """
    Fully specified optimization configuration.
    - terms: list of objective terms; if len>1 --> multi-objective (unless combine='weighted_sum')
    - combine: 'auto', 'weighted_sum', or 'pareto'
    - constraints: list of inequality constraints (<=0 feasible)
    - cycles: number of simulation cycles (default 3)
    - dt: time resolution
    """
    terms: List[ObjectiveTerm] = field(default_factory=list)
    combine: str = "auto"                        # 'auto' uses 'pareto' for >1 term else single-objective
    constraints: List[Constraint] = field(default_factory=list)
    cycles: int = 3
    dt: float = 0.05


class ObjectiveManager:
    """
    Translates natural language to objective functions and constraints.
    Also provides helpers to evaluate objectives on a given (x, prediction).
    """

    def __init__(self, model: ModelInterface):
        self.model = model

    # ---------- Parsing ----------
    def parse_objective(self, user_input: str) -> OptimizationConfig:
        """
        Heuristic NL parser for common patterns and constraints.
        """
        text = user_input.lower().strip()
        cfg = OptimizationConfig(cycles=3, dt=self.model.default_dt)

        # Multi-objective keywords
        if "pareto" in text or "multi-objective" in text:
            cfg.combine = "pareto"

        # Basic single metrics
        if "minimize cycle time" in text or "minimum cycle time" in text:
            cfg.terms.append(ObjectiveTerm("cycle_time", Direction.MIN))
        if "minimize on-time" in text or "minimum on-time" in text or "minimize magnetic on-time" in text:
            cfg.terms.append(ObjectiveTerm("on_time", Direction.MIN))
        if "minimize off-time" in text or "minimum off-time" in text or "minimize magnetic off-time" in text:
            cfg.terms.append(ObjectiveTerm("off_time", Direction.MIN))
        if "maximize" in text and "segregation" in text:
            cfg.terms.append(ObjectiveTerm("mean_seg", Direction.MAX))
        if "minimize" in text and "segregation" in text:
            cfg.terms.append(ObjectiveTerm("mean_seg", Direction.MIN))

        # Sustained segregation or mixing constraints
        # Examples:
        # "Segregation index > 0.9 during on-cycle for at least 2 seconds"
        # "Mixing (segregation < 0.1) during off-cycle for at least 1.5 seconds"
        seg_high = re.search(r"segregation\s*(index)?\s*>\s*([0-9]*\.?[0-9]+)", text)
        seg_low = re.search(r"segregation\s*(index)?\s*<\s*([0-9]*\.?[0-9]+)", text)
        dur = re.search(r"(at least|>=?)\s*([0-9]*\.?[0-9]+)\s*(s|sec|seconds)", text)

        wants_on = "on-cycle" in text or "during on" in text or "when on" in text
        wants_off = "off-cycle" in text or "during off" in text or "when off" in text
        sustained_all = "all 3 cycles" in text or "across all 3 cycles" in text or "sustained performance" in text

        if seg_high and dur and wants_on:
            thr = float(seg_high.group(2))
            req = float(dur.group(2))
            cfg.constraints.append(self._constraint_high_on(threshold=thr, seconds=req, sustained=sustained_all))

        if seg_low and dur and wants_off:
            thr = float(seg_low.group(2))
            req = float(dur.group(2))
            cfg.constraints.append(self._constraint_low_off(threshold=thr, seconds=req, sustained=sustained_all))

        # Defaults if nothing specific captured
        if not cfg.terms:
            # common default: Pareto between mean and peak segregation (minimize both)
            cfg.terms.extend([ObjectiveTerm("mean_seg", Direction.MIN), ObjectiveTerm("peak_seg", Direction.MIN)])
            cfg.combine = "pareto"

        return cfg

    # ---------- Objective & Constraint Builders ----------
    def create_objective_function(self, cfg: OptimizationConfig):
        """
        Returns:
          n_obj: int
          f: callable(X)-> F (vectorized over rows)
          g: callable(X)-> G (vectorized constraints; <=0 feasible) or None
        """
        # Determine number of objectives
        if cfg.combine == "weighted_sum" and len(cfg.terms) > 1:
            n_obj = 1
        else:
            n_obj = len(cfg.terms)

        # Objective f(X)
        def f(X: np.ndarray) -> np.ndarray:
            # X shape: (N, 3) as [on, off, vel]
            if X.ndim == 1:
                X2 = X.reshape(1, -1)
            else:
                X2 = X

            # Build parameter_array for batch predict (N, 3)
            preds = self.model.batch_predict(X2)
            # Compute metrics
            vals = []
            for i in range(X2.shape[0]):
                on, off, vel = X2[i, 0], X2[i, 1], X2[i, 2]
                cyc = on + off
                t = preds[i]["t"]
                s = preds[i]["seg_index"]
                metrics = self._metrics(on, off, vel, cyc, t, s)
                if n_obj == 1 and len(cfg.terms) > 1 and cfg.combine == "weighted_sum":
                    ws = 0.0
                    for term in cfg.terms:
                        ws += term.weight * self._term_value(term, metrics)
                    vals.append([ws])
                else:
                    vec = [self._term_value(term, metrics) for term in cfg.terms]
                    vals.append(vec)
            V = np.asarray(vals, dtype=float)

            # Convert MAX to MIN by negating those columns
            for j, term in enumerate([cfg.terms] if n_obj == 1 and len(cfg.terms) == 1 else cfg.terms):
                # if weighted sum, handled in _term_value already (direction)
                pass
            # If multiple objectives, negate columns where direction == MAX
            if n_obj > 1:
                for j, term in enumerate(cfg.terms):
                    if term.direction == Direction.MAX:
                        V[:, j] = -V[:, j]
            else:
                # Single objective with single term
                if len(cfg.terms) == 1 and cfg.terms[0].direction == Direction.MAX:
                    V[:, 0] = -V[:, 0]
            return V

        # Constraints g(X)
        def g(X: np.ndarray) -> Optional[np.ndarray]:
            if not cfg.constraints:
                return None
            if X.ndim == 1:
                X2 = X.reshape(1, -1)
            else:
                X2 = X
            preds = self.model.batch_predict(X2)
            G_rows = []
            for i in range(X2.shape[0]):
                on, off, vel = X2[i, 0], X2[i, 1], X2[i, 2]
                cyc = on + off
                t = preds[i]["t"]
                s = preds[i]["seg_index"]
                pred_dict = {"t": t, "seg": s, "on_time": on, "off_time": off, "cycle_time": cyc}
                gvals = []
                for c in cfg.constraints:
                    try:
                        gv = np.atleast_1d(c.func(X2[i], pred_dict)).astype(float)
                    except Exception as e:
                        logger.error("Constraint evaluation failed (%s). Applying penalty.", e)
                        gv = np.array([1e3], dtype=float)
                    gvals.extend(gv.tolist())
                G_rows.append(gvals)
            return np.asarray(G_rows, dtype=float)

        return n_obj, f, g

    def add_constraint(self, cfg: OptimizationConfig, constraint_description: str):
        """Parse and add a constraint to an existing config."""
        add_cfg = self.parse_objective(constraint_description)
        cfg.constraints.extend(add_cfg.constraints)

    def evaluate_objectives(self, parameters: np.ndarray, model_predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute a standard set of metrics for a given (x, pred)."""
        on, off, vel = parameters[0], parameters[1], parameters[2]
        cyc = on + off
        t, s = model_predictions["t"], model_predictions["seg_index"]
        return self._metrics(on, off, vel, cyc, t, s)

    # ---------- Metric helpers ----------
    @staticmethod
    def _metrics(on: float, off: float, vel: float, cyc: float, t: np.ndarray, s: np.ndarray) -> Dict[str, float]:
        mean_seg = float(np.mean(s))
        peak_seg = float(np.max(s))
        final_seg = float(s[-1])
        return {
            "on_time": on,
            "off_time": off,
            "velocity": vel,
            "cycle_time": cyc,
            "mean_seg": mean_seg,
            "peak_seg": peak_seg,
            "final_seg": final_seg,
        }

    @staticmethod
    def _term_value(term: ObjectiveTerm, metrics: Dict[str, float]) -> float:
        val = float(metrics.get(term.name, np.nan))
        if np.isnan(val):
            raise ValueError(f"Unknown metric in objective: {term.name}")
        # For weighted sum, encode direction via sign
        if term.direction == Direction.MAX:
            return -val * term.weight
        return val * term.weight

    # ---------- Constraint builders ----------
    def _constraint_high_on(self, threshold: float, seconds: float, sustained: bool) -> Constraint:
        desc = f"Segregation >= {threshold:.3f} for >= {seconds:.2f}s during on-cycle" + (" in each cycle" if sustained else "")

        def g(x: np.ndarray, pred: Dict[str, np.ndarray]) -> np.ndarray:
            t, s = pred["t"], pred["seg"]
            on = pred["on_time"]; off = pred["off_time"]; period = on + off
            # measure time within on-phase where seg >= threshold per cycle
            violations = []
            total_cycles = int(np.floor(t[-1] / period + 1e-9))
            # segment time steps
            for c in range(total_cycles):
                mask_cycle = (t >= c * period) & (t <= (c + 1) * period + 1e-9)
                t_c = t[mask_cycle]
                s_c = s[mask_cycle]
                t_in = (t_c - c * period) % period
                mask_on = t_in <= on
                if t_c.size <= 1:
                    dt = 0.0
                else:
                    dt = float(np.median(np.diff(t_c)))
                duration = np.sum((s_c[mask_on] >= threshold).astype(float)) * dt
                if sustained:
                    violations.append(max(0.0, seconds - duration))
                else:
                    # if not sustained, we only need one cycle to satisfy
                    violations.append(seconds - duration)
            if sustained:
                # require all cycles satisfied: return vector of per-cycle violations
                return np.array(violations, dtype=float)
            # otherwise single scalar: min over cycles should be >=0 → violation = max(0, req - max_achieved)
            return np.array([max(0.0, max(violations))], dtype=float)
        return Constraint(description=desc, func=g, weight=10.0)

    def _constraint_low_off(self, threshold: float, seconds: float, sustained: bool) -> Constraint:
        desc = f"Segregation <= {threshold:.3f} for >= {seconds:.2f}s during off-cycle" + (" in each cycle" if sustained else "")

        def g(x: np.ndarray, pred: Dict[str, np.ndarray]) -> np.ndarray:
            t, s = pred["t"], pred["seg"]
            on = pred["on_time"]; off = pred["off_time"]; period = on + off
            violations = []
            total_cycles = int(np.floor(t[-1] / period + 1e-9))
            for c in range(total_cycles):
                mask_cycle = (t >= c * period) & (t <= (c + 1) * period + 1e-9)
                t_c = t[mask_cycle]; s_c = s[mask_cycle]
                t_in = (t_c - c * period) % period
                mask_off = t_in > on
                dt = float(np.median(np.diff(t_c))) if t_c.size > 1 else 0.0
                duration = np.sum((s_c[mask_off] <= threshold).astype(float)) * dt
                if sustained:
                    violations.append(max(0.0, seconds - duration))
                else:
                    violations.append(seconds - duration)
            if sustained:
                return np.array(violations, dtype=float)
            return np.array([max(0.0, max(violations))], dtype=float)
        return Constraint(description=desc, func=g, weight=10.0)
