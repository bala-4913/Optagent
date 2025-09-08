# mfb_opt/optimization.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import BOUNDS, RANDOM_SEED
from .utils.logging_utils import get_logger

logger = get_logger("MFB.OptimizationEngine")

# ---- pymoo imports (guarded) ----
try:
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.algorithms.moo.moead import MOEAD
    from pymoo.algorithms.soo.nonconvex.ga import GA
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
    from pymoo.algorithms.soo.nonconvex.pso import PSO
    from pymoo.termination import get_termination
    from pymoo.util.ref_dirs import get_reference_directions
    from pymoo.core.repair import Repair
    from pymoo.core.sampling import Sampling
    from pymoo.core.mutation import Mutation
    from pymoo.core.crossover import Crossover
except Exception as e:
    Problem = object
    NSGA2 = NSGA3 = MOEAD = GA = DE = CMAES = PSO = None
    minimize = None
    get_termination = None
    get_reference_directions = None
    Repair = Sampling = Mutation = Crossover = object
    logger.warning("pymoo not fully available. Please `pip install pymoo`.")


@dataclass
class OptimizationResult:
    X: np.ndarray
    F: np.ndarray
    G: Optional[np.ndarray]
    history: Dict[str, Any]
    algorithm: str
    success: bool


class BoundRepair(Repair):
    """Simple repair to clip decision variables into bounds."""
    def __init__(self, xl: np.ndarray, xu: np.ndarray):
        super().__init__()
        self.xl = xl
        self.xu = xu

    def _do(self, problem, X, **kwargs):
        return np.clip(X, self.xl, self.xu)


class MFBProblem(Problem):
    def __init__(self, n_obj: int, f, g=None):
        xl = np.array([BOUNDS["on_time"][0], BOUNDS["off_time"][0], BOUNDS["velocity"][0]], dtype=float)
        xu = np.array([BOUNDS["on_time"][1], BOUNDS["off_time"][1], BOUNDS["velocity"][1]], dtype=float)
        self._f = f
        self._g = g

        n_constr = 0
        if g is not None:
            # probe constraints length
            try:
                probe = np.array([(xl + xu) / 2.0])
                g_val = g(probe)
                n_constr = int(g_val.shape[1]) if g_val is not None and g_val.ndim == 2 else (int(len(g_val[0])) if g_val is not None else 0)
            except Exception:
                n_constr = 0

        super().__init__(n_var=3, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        try:
            F = self._f(X)
        except Exception as e:
            logger.exception("Objective eval failed. Penalizing. %s", e)
            if isinstance(X, np.ndarray) and X.ndim == 2:
                N = X.shape[0]
            else:
                N = 1
            F = np.full((N, self.n_obj), 1e6, dtype=float)
        out["F"] = F

        if self.n_constr > 0 and self._g is not None:
            try:
                G = self._g(X)
            except Exception as e:
                logger.exception("Constraint eval failed. Applying penalty. %s", e)
                G = np.full((X.shape[0], self.n_constr), 1e3, dtype=float)
            out["G"] = G


class OptimizationEngine:
    """
    Wraps pymoo algorithms with smart defaults and utilities.
    """

    def __init__(self, seed: int = RANDOM_SEED):
        if minimize is None:
            raise RuntimeError("pymoo not available. Install with `pip install pymoo`.")
        self.seed = seed
        self._algorithm = None
        self._termination = get_termination("n_gen", 60)
        self._history = {}
        self._last_result: Optional[OptimizationResult] = None

    # ----- Algorithm selection -----
    def set_algorithm(self, algorithm_name: str = "auto", n_obj: int = 1, pop_size: Optional[int] = None, **params):
        alg = (algorithm_name or "auto").lower()
        if pop_size is None:
            pop_size = self._default_pop_size(n_obj)

        # Repair to enforce bounds strictly
        xl = np.array([BOUNDS["on_time"][0], BOUNDS["off_time"][0], BOUNDS["velocity"][0]], dtype=float)
        xu = np.array([BOUNDS["on_time"][1], BOUNDS["off_time"][1], BOUNDS["velocity"][1]], dtype=float)
        repair = BoundRepair(xl, xu)

        if alg == "auto":
            if n_obj > 1:
                self._algorithm = NSGA2(pop_size=pop_size, repair=repair, **params)
            else:
                self._algorithm = GA(pop_size=pop_size, repair=repair, **params)
        elif alg in ("ga", "genetic"):
            self._algorithm = GA(pop_size=pop_size, repair=repair, **params)
        elif alg in ("de", "differential_evolution"):
            self._algorithm = DE(pop_size=pop_size, repair=repair, **params)
        elif alg in ("pso",):
            self._algorithm = PSO(pop_size=pop_size, **params)
        elif alg in ("cmaes", "cma-es"):
            self._algorithm = CMAES(pop_size=pop_size, **params)
        elif alg in ("nsga2", "nsga-ii"):
            self._algorithm = NSGA2(pop_size=pop_size, repair=repair, **params)
        elif alg in ("nsga3", "nsga-iii"):
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            self._algorithm = NSGA3(pop_size=max(pop_size, len(ref_dirs)), ref_dirs=ref_dirs, repair=repair, **params)
        elif alg in ("moead", "moea/d"):
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            self._algorithm = MOEAD(ref_dirs=ref_dirs, **params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        logger.info("Selected algorithm: %s (pop=%d)", self._algorithm.__class__.__name__, pop_size)

    def add_termination_criteria(self, max_gen: int = 60, target_fitness: Optional[float] = None, f_tol: Optional[float] = 1e-4, n_last: int = 15):
        # Combine generation count and improvement tolerance if available
        term = get_termination("n_gen", max_gen)
        if f_tol is not None:
            try:
                from pymoo.termination.default import DefaultMultiObjectiveTermination
                # For MO problems; pymoo uses DefaultMultiObjectiveTermination combining several criteria
                # We'll keep basic setup; for SO, f_tol not directly combined here
            except Exception:
                pass
        self._termination = term

    # ----- Optimize -----
    def optimize(self, objective_function, n_vars: int = 3, bounds: Dict[str, Tuple[float, float]] = BOUNDS,
                 algorithm: str = "auto") -> OptimizationResult:
        n_obj, f, g = objective_function
        self.set_algorithm(algorithm_name=algorithm, n_obj=n_obj)

        problem = MFBProblem(n_obj=n_obj, f=f, g=g)
        logger.info("Starting optimization: n_obj=%d", n_obj)

        res = minimize(
            problem=problem,
            algorithm=self._algorithm,
            termination=self._termination,
            seed=self.seed,
            save_history=True,
            verbose=False,
        )

        X = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        G = np.atleast_2d(res.G) if hasattr(res, "G") and res.G is not None else None

        hist = {
            "n_evals": getattr(res, "n_evals", None),
            "algorithm": self._algorithm.__class__.__name__,
            "history": [getattr(e, "opt", None) for e in getattr(res, "history", [])],
            "cv": getattr(res, "CV", None),  # constraint violation history if available
        }

        result = OptimizationResult(
            X=X, F=F, G=G, history=hist, algorithm=self._algorithm.__class__.__name__, success=True
        )
        self._last_result = result
        return result

    def get_pareto_front(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not self._last_result:
            return None
        return self._last_result.X, self._last_result.F

    def get_optimization_history(self) -> Dict[str, Any]:
        return self._last_result.history if self._last_result else {}

    # ----- Helpers -----
    @staticmethod
    def _default_pop_size(n_obj: int) -> int:
        return 60 if n_obj <= 1 else 120
