# mfb_opt/agent.py
from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np

from .model_interface import ModelInterface
from .objectives import ObjectiveManager, OptimizationConfig
from .optimization import OptimizationEngine
from .analysis import ResultsAnalyzer
from .utils.logging_utils import get_logger

logger = get_logger("MFB.ConversationAgent")


class ConversationAgent:
    def __init__(self, model: ModelInterface):
        self.model = model
        self.om = ObjectiveManager(model)
        self.engine = OptimizationEngine()
        self.analyzer = ResultsAnalyzer(model)

    # ---- Conversation flow ----
    def process_user_request(self, user_input: str) -> Dict[str, Any]:
        # Parse request → config
        cfg = self.om.parse_objective(user_input)
        return {"config": cfg}

    def ask_clarification(self, ambiguous_request: str) -> str:
        # Simple heuristics
        if "segregation" in ambiguous_request.lower() and "minimize" not in ambiguous_request.lower() and "maximize" not in ambiguous_request.lower():
            return "Do you want to minimize average segregation, minimize peak, or maximize segregation during on-cycle?"
        return "Could you clarify whether this should be single-objective or a Pareto trade-off?"

    def run(self, cfg: OptimizationConfig, algorithm: str = "auto"):
        n_obj, f, g = self.om.create_objective_function(cfg)
        res = self.engine.optimize((n_obj, f, g), algorithm=algorithm)
        return res

    def explain_results(self, results) -> str:
        X, F = results.X, results.F
        if F.shape[1] == 1:
            idx = int(np.argmin(F[:, 0]))
            x = X[idx]
            return (f"Optimal parameters (single-objective): on={x[0]:.2f}s, off={x[1]:.2f}s, v={x[2]:.2f} m/s. "
                    f"Objective={F[idx,0]:.4f}")
        else:
            msg = [f"Found {X.shape[0]} Pareto-optimal candidates. First 5:"]
            K = min(5, X.shape[0])
            for i in range(K):
                msg.append(f"  #{i+1}: on={X[i,0]:.2f}s, off={X[i,1]:.2f}s, v={X[i,2]:.2f} → F={F[i].tolist()}")
            return "\n".join(msg)

    def recommend_parameters(self, results, user_priorities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.analyzer.recommend_solution(results, user_priorities)
