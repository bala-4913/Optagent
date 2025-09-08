# mfb_opt/main.py
from __future__ import annotations

import json
import argparse
from pathlib import Path

from .model_interface import ModelInterface
from .agent import ConversationAgent
from .analysis import ResultsAnalyzer
from .utils.logging_utils import get_logger

logger = get_logger("MFB.Main")


def run_cli():
    parser = argparse.ArgumentParser(description="MFB Optimization System")
    parser.add_argument("--request", type=str, required=False, default="Find a Pareto trade-off between mean and peak segregation.")
    parser.add_argument("--algorithm", type=str, required=False, default="auto")
    parser.add_argument("--config", type=str, required=False, help="Path to JSON config with request and algorithm.")
    parser.add_argument("--outdir", type=str, required=False, default="outputs")
    args = parser.parse_args()

    if args.config:
        cfg_path = Path(args.config)
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        request = cfg.get("request", args.request)
        algorithm = cfg.get("algorithm", args.algorithm)
    else:
        request = args.request
        algorithm = args.algorithm

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # 1) Load model
    model = ModelInterface()
    logger.info("Model info: %s", model.get_model_info())

    # 2) Agent
    agent = ConversationAgent(model)

    # 3) Parse + run optimization
    parsed = agent.process_user_request(request)
    cfg = parsed["config"]
    results = agent.run(cfg, algorithm=algorithm)
    logger.info(agent.explain_results(results))

    # 4) Analyze + report
    analyzer = ResultsAnalyzer(model)
    report_pdf = str(Path(args.outdir) / "report.pdf")
    csv_out = str(Path(args.outdir) / "solutions.csv")
    analyzer.generate_report(results, request, save_pdf_path=report_pdf, export_csv_path=csv_out)
    logger.info("Done. Outputs saved to %s", args.outdir)


if __name__ == "__main__":
    run_cli()
