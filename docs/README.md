# MFB Optimization System

An intelligent optimization agent for Magnetic Fluidized Bed (MFB) processes.

## Features
- **ModelInterface**: Fast, cached predictions using a pre-trained FFNN (`best_ffnn_model.h5`) with scalers.
- **ObjectiveManager**: Natural language objectives → mathematical functions with time-based constraints.
- **OptimizationEngine**: Efficient single/multi-objective algorithms (GA, DE, PSO, CMA-ES, NSGA-II/III, MOEA/D).
- **ResultsAnalyzer**: Pareto, sensitivity, convergence, and reporting (PDF/CSV).
- **ConversationAgent**: Natural language interface and explanations.

## Installation
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
