# mfb_opt/analysis.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px

from .model_interface import ModelInterface
from .utils.logging_utils import get_logger

logger = get_logger("MFB.ResultsAnalyzer")


@dataclass
class ResultsAnalyzer:
    model: ModelInterface

    # ---- Summaries ----
    def analyze_results(self, optimization_results) -> Dict[str, Any]:
        X, F = optimization_results.X, optimization_results.F
        summary = {
            "n_solutions": X.shape[0],
            "objective_dim": F.shape[1],
            "X_stats": pd.DataFrame(X, columns=["on_time", "off_time", "velocity"]).describe().to_dict(),
            "F_stats": pd.DataFrame(F, columns=[f"obj{i+1}" for i in range(F.shape[1])]).describe().to_dict(),
        }
        return summary

    # ---- Pareto ----
    def plot_pareto_front(self, results, save_path: Optional[str] = None):
        X, F = results.X, results.F
        if F.shape[1] == 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(range(len(F)), F[:, 0], c="tab:blue", s=25)
            ax.set_xlabel("Solution index"); ax.set_ylabel("Objective value (min)")
            ax.grid(True); plt.tight_layout()
        elif F.shape[1] == 2:
            fig = px.scatter(x=F[:, 0], y=F[:, 1], labels={"x": "Objective 1", "y": "Objective 2"},
                             title="Pareto Front (2D)")
            if save_path: fig.write_html(save_path)
            return fig
        else:
            fig = px.scatter_3d(x=F[:, 0], y=F[:, 1], z=F[:, 2],
                                labels={"x": "Obj1", "y": "Obj2", "z": "Obj3"},
                                title="Pareto Front (3D)")
            if save_path: fig.write_html(save_path)
            return fig

        if save_path:
            plt.savefig(save_path, dpi=160, bbox_inches="tight")
        return fig

    # ---- Sensitivity ----
    def sensitivity_analysis(self, solution: np.ndarray, eps: Dict[str, float] = None) -> pd.DataFrame:
        if eps is None: eps = {"on_time": 0.1, "off_time": 0.1, "velocity": 0.01}
        base = self._metrics_for(solution)
        grads = []
        for i, k in enumerate(["on_time", "off_time", "velocity"]):
            x_pert = solution.copy()
            x_pert[i] += eps[k]
            pert = self._metrics_for(x_pert)
            grads.append({
                "param": k,
                "d_mean_seg": (pert["mean_seg"] - base["mean_seg"]) / eps[k],
                "d_peak_seg": (pert["peak_seg"] - base["peak_seg"]) / eps[k],
                "d_final_seg": (pert["final_seg"] - base["final_seg"]) / eps[k],
                "d_cycle_time": (pert["cycle_time"] - base["cycle_time"]) / eps[k],
            })
        return pd.DataFrame(grads)

    def _metrics_for(self, x: np.ndarray) -> Dict[str, float]:
        on, off, vel = x
        cyc = on + off
        pred = self.model.predict_segregation_trajectory(on, off, vel)
        s = pred["seg_index"]; t = pred["t"]
        return {
            "mean_seg": float(np.mean(s)),
            "peak_seg": float(np.max(s)),
            "final_seg": float(s[-1]),
            "cycle_time": float(cyc),
        }

    # ---- Convergence ----
    def plot_convergence(self, history: Dict[str, Any]):
        # Minimal placeholder: number of evaluations (if available)
        n_evals = history.get("n_evals", None)
        if n_evals is None:
            logger.info("No convergence history available to plot.")
            return None
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(n_evals, marker="o")
        ax.set_title("Convergence (n_evals over time)")
        ax.set_xlabel("Checkpoint"); ax.set_ylabel("Evaluations")
        ax.grid(True); plt.tight_layout()
        return fig

    # ---- Reports & exports ----
    def generate_report(self, results, user_objectives: str, save_pdf_path: str = "mfb_report.pdf",
                        export_csv_path: Optional[str] = "mfb_solutions.csv"):
        X, F = results.X, results.F
        # Save CSV
        if export_csv_path:
            df = pd.DataFrame(X, columns=["on_time", "off_time", "velocity"])
            for j in range(F.shape[1]):
                df[f"obj{j+1}"] = F[:, j]
            df.to_csv(export_csv_path, index=False)
            logger.info("Exported solutions to %s", export_csv_path)

        # Build PDF with plots
        with PdfPages(save_pdf_path) as pdf:
            # Pareto
            if F.shape[1] == 2:
                fig = self.plot_pareto_front(results)
                fig.update_layout(title=f"Pareto Front — {user_objectives}")
                # export static PNG into PDF
                fig.write_image("pareto_tmp.png", scale=2)  # requires kaleido; if unavailable, fallback to mpl below
                img = plt.imread("pareto_tmp.png")
                plt.figure(figsize=(6, 5)); plt.imshow(img); plt.axis("off"); pdf.savefig(); plt.close()
            else:
                fig = self.plot_pareto_front(results)
                pdf.savefig(fig); plt.close(fig)

            # Sensitivity for first solution
            sens = self.sensitivity_analysis(results.X[0])
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sens.set_index("param")[["d_mean_seg", "d_peak_seg", "d_final_seg"]].plot(kind="bar", ax=ax2)
            ax2.set_title("Local Sensitivity (Solution #1)"); ax2.grid(True); plt.tight_layout()
            pdf.savefig(fig2); plt.close(fig2)

            # Text page
            fig3 = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            txt = f"User objectives:\n{user_objectives}\n\nSolutions saved to: {export_csv_path}"
            plt.text(0.05, 0.95, txt, va="top", fontsize=12)
            plt.axis("off"); pdf.savefig(fig3); plt.close(fig3)

        logger.info("Saved PDF report to %s", save_pdf_path)

    def recommend_solution(self, results, user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Choose a solution based on simple rules or preferences.
        If single objective: min F. If multi: pick knee-point (approx via min sum normalized).
        """
        X, F = results.X, results.F
        if F.shape[1] == 1:
            idx = int(np.argmin(F[:, 0]))
            return {"index": idx, "x": X[idx], "f": F[idx]}
        # multi-objective: min of L2 distance from ideal (min each column)
        f_min = F.min(axis=0); f_max = F.max(axis=0); denom = np.where(f_max > f_min, f_max - f_min, 1.0)
        scores = np.linalg.norm((F - f_min) / denom, axis=1)
        idx = int(np.argmin(scores))
        return {"index": idx, "x": X[idx], "f": F[idx]}
