import sys
from typing import Dict, List, Any

import numpy as np
import torch

from hypermapper.bo.models import models
from hypermapper.param.space import Space


def ucb(
    settings: Dict,
    param_space: Space,
    X: torch.Tensor,
    objective_weights: torch.Tensor,
    regression_models: List[Any],
    iteration_number: int,
    classification_model: Any,
    feasibility_threshold,
    **kwargs,
) -> torch.Tensor:
    """
    Multi-objective ucb acquisition function as detailed in https://arxiv.org/abs/1805.12168.
    The mean and variance of the predictions are computed as defined by Hutter et al.: https://arxiv.org/pdf/1211.0906.pdf

    Input:
        - settings: the Hypermapper run settings
        - param_space: a space object containing the search space.
        - X: a list of tuples containing the points to predict and scalarize.
        - objective_weights: a list containing the weights for each objective.
        - regression_models: the surrogate models used to evaluate points.
        - iteration_number: an integer for the current iteration number, used to compute the beta
        - classification_model: the surrogate model used to evaluate feasibility constraints
        - feasibility_threshold: minimum probability of feasibility
        - kwargs: to throw away additional input
    Returns:
        - a tensor of scalarized values for each point in X.
    """
    pass


def ei(
    settings: dict,
    param_space: Space,
    X: torch.Tensor,
    objective_weights: List[float],
    regression_models: List[Any],
    best_values: float,
    objective_means: torch.Tensor,
    objective_stds: torch.Tensor,
    classification_model: Any,
    feasibility_threshold: float,
    verbose: bool = False,
    **kwargs,
) -> torch.Tensor:
    pass