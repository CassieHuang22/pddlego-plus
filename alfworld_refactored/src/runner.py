from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from .pddl_engine import (
    run_iterative_model as _run_translator,
    run_baseline_model as _run_direct_planner,
    run_iterative_model_initDF as _run_translator_seed_df,
)

class Method(Enum):
    DIRECT_PLANNER = "direct_planner"
    TRANSLATOR = "translator"
    TRANSLATOR_WITH_SEED_DF = "translator_seed_df"

@dataclass
class ExperimentConfig:

    model_name: str
    start_trial: int = 0
    end_trial: int = 10
    folder_name: str = "alfworld_runs"
    result_name: str = "alfworld_results"
    goal_type: str = "detailed"
    seed_df: str | None = None

def run_translator(cfg: ExperimentConfig) -> None:
    _run_translator(
        cfg.model_name,
        cfg.start_trial,
        cfg.end_trial,
        cfg.folder_name,
        cfg.result_name,
        cfg.goal_type,
    )

def run_translator_with_seed_df(cfg: ExperimentConfig) -> None:
    if not cfg.seed_df:
        raise ValueError("cfg.seed_df is required for TRANSLATOR_WITH_SEED_DF")
    _run_translator_seed_df(
        cfg.model_name,
        cfg.start_trial,
        cfg.end_trial,
        cfg.folder_name,
        cfg.result_name,
        cfg.seed_df,
        cfg.goal_type,
        "initDF",
    )

def run_direct_planner(cfg: ExperimentConfig) -> None:
    _run_direct_planner(cfg.model_name, cfg.start_trial, cfg.end_trial, cfg.folder_name, cfg.result_name)

def run(cfg: ExperimentConfig, method: Method) -> None:
    if method is Method.TRANSLATOR:
        return run_translator(cfg)
    if method is Method.TRANSLATOR_WITH_SEED_DF:
        return run_translator_with_seed_df(cfg)
    if method is Method.DIRECT_PLANNER:
        return run_direct_planner(cfg)
    raise ValueError(f"Unknown method: {method}")
