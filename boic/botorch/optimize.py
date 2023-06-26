from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor

import botorch
from botorch.acquisition.acquisition import OneShotAcquisitionFunction

from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

# from botorch.generation.gen import gen_candidates_scipy
from .generate import gen_candidates_scipy # has precision argument

from botorch.logging import logger
from botorch.optim.initializers import (
    #gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion

from botorch import settings
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    _get_value_function,
    qKnowledgeGradient,
)
from botorch.acquisition.utils import is_nonnegative
from botorch.exceptions.warnings import BadInitialCandidatesWarning, SamplingWarning
from botorch.models.model import Model
from botorch.optim.utils import fix_features
from botorch.utils.sampling import (
    batched_multinomial,
    draw_sobol_samples,
    manual_seed,
    get_polytope_samples,
)
from botorch.utils.transforms import standardize
from torch import Tensor
from torch.quasirandom import SobolEngine


# Note: adding seed for reproducibility in gen_batch_initial_conditions
def optimize_acqf(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:
    r"""Generate a set of candidates via multi-start optimization.

    Args:
        acq_function: An AcquisitionFunction.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of samples for initialization. This is required
            if `batch_initial_conditions` is not specified.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        return_best_only: If False, outputs the solutions corresponding to all
            random restart initializations of the optimization.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
        kwargs: Additional keyword arguments.

    Returns:
        A two-element tuple containing

        - a `(num_restarts) x q x d`-dim tensor of generated candidates.
        - a tensor of associated acquisition values. If `sequential=False`,
            this is a `(num_restarts)`-dim tensor of joint acquisition values
            (with explicit restart dimension if `return_best_only=False`). If
            `sequential=True`, this is a `q`-dim tensor of expected acquisition
            values conditional on having observed candidates `0,1,...,i-1`.
    """
    if sequential and q > 1:
        if not return_best_only:
            raise NotImplementedError(
                "`return_best_only=False` only supported for joint optimization."
            )
        if isinstance(acq_function, OneShotAcquisitionFunction):
            raise NotImplementedError(
                "sequential optimization currently not supported for one-shot "
                "acquisition functions. Must have `sequential=False`."
            )
        candidate_list, acq_value_list = [], []
        base_X_pending = acq_function.X_pending
        for i in range(q):
            candidate, acq_value = optimize_acqf(
                acq_function=acq_function,
                bounds=bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options=options or {},
                inequality_constraints=inequality_constraints,
                equality_constraints=equality_constraints,
                fixed_features=fixed_features,
                post_processing_func=post_processing_func,
                batch_initial_conditions=None,
                return_best_only=True,
                sequential=False,
            )
            candidate_list.append(candidate)
            acq_value_list.append(acq_value)
            candidates = torch.cat(candidate_list, dim=-2)
            acq_function.set_X_pending(
                torch.cat([base_X_pending, candidates], dim=-2)
                if base_X_pending is not None
                else candidates
            )
            logger.info(f"Generated sequential candidate {i+1} of {q}")
        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(base_X_pending)
        return candidates, torch.stack(acq_value_list)

    options = options or {}

    # Handle the trivial case when all features are fixed
    if fixed_features is not None and len(fixed_features) == bounds.shape[-1]:
        X = torch.tensor(
            [fixed_features[i] for i in range(bounds.shape[-1])],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        X = X.expand(q, *X.shape)
        with torch.no_grad():
            acq_value = acq_function(X)
        return X, acq_value

    if batch_initial_conditions is None:
        if raw_samples is None:
            raise ValueError(
                "Must specify `raw_samples` when `batch_initial_conditions` is `None`."
            )

        ic_gen = (
            gen_one_shot_kg_initial_conditions
            if isinstance(acq_function, qKnowledgeGradient)
            else gen_batch_initial_conditions
        )
        batch_initial_conditions = ic_gen(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            **({'additional_samples': kwargs['additional_samples']} if 'additional_samples' in kwargs
               else {}))
    batch_limit: int = options.get("batch_limit", num_restarts)
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    batched_ics = batch_initial_conditions.split(batch_limit)
    for i, batched_ics_ in enumerate(batched_ics):
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
            initial_conditions=batched_ics_,
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={
                k: v
                for k, v in options.items()
                if k not in ("init_batch_limit", "batch_limit", "nonnegative")
            },
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            precision = kwargs.get('precision')
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values


def gen_batch_initial_conditions(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    additional_samples: Optional[Tensor] = None
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimziation.

    TODO: Support t-batches of initial conditions.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.
    """
    options = options or {}
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = botorch.optim.initializers.initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = botorch.optim.initializers.initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor
            if inequality_constraints is None and equality_constraints is None:
                if effective_dim <= SobolEngine.MAXDIM:
                    X_rnd = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed)
                else:
                    with manual_seed(seed):
                        # load on cpu
                        X_rnd_nlzd = torch.rand(
                            n, q, bounds.shape[-1], dtype=bounds.dtype
                        )
                    X_rnd = bounds[0] + (bounds[1] - bounds[0]) * X_rnd_nlzd
            else:
                X_rnd = (
                    get_polytope_samples(
                        n=n * q,
                        bounds=bounds,
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                        seed=seed,
                        n_burnin=options.get("n_burnin", 10000),
                        thinning=options.get("thinning", 32),
                    )
                    .view(n, q, -1)
                    .cpu()
                )
            if additional_samples is not None:
                X_rnd = torch.cat((additional_samples, X_rnd), dim=0)
            X_rnd = fix_features(X_rnd, fixed_features=fixed_features)
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            # print(Y_rnd.argsort(descending=True)[:10])
            # print(Y_rnd[Y_rnd.argsort(descending=True)[:10]])
            # print(X_rnd[Y_rnd.argsort(descending=True)[:10]])
            # print(Y_rnd[:10])
            # print(X_rnd[:10])
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(issubclass(w.category, BadInitialCandidatesWarning) for w in ws):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions















def optimize_acqf2(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: Optional[int] = None,
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    fixed_features: Optional[Dict[int, float]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    return_best_only: bool = True,
    sequential: bool = False,
    **kwargs: Any,
) -> Tuple[Tensor, Tensor]:

    options = options or {}

    # Handle the trivial case when all features are fixed
    if fixed_features is not None and len(fixed_features) == bounds.shape[-1]:
        X = torch.tensor(
            [fixed_features[i] for i in range(bounds.shape[-1])],
            device=bounds.device,
            dtype=bounds.dtype,
        )
        X = X.expand(q, *X.shape)
        with torch.no_grad():
            acq_value = acq_function(X)
        return X, acq_value

    if batch_initial_conditions is None:
        if raw_samples is None:
            raise ValueError(
                "Must specify `raw_samples` when `batch_initial_conditions` is `None`."
            )

        batch_initial_conditions, X_rnd, Y_rnd = gen_batch_initial_conditions2(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            fixed_features=fixed_features,
            options=options,
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            **({'additional_samples': kwargs['additional_samples']} if 'additional_samples' in kwargs
               else {}))
    batch_limit: int = options.get("batch_limit", num_restarts)
    batch_candidates_list: List[Tensor] = []
    batch_acq_values_list: List[Tensor] = []
    batched_ics = batch_initial_conditions.split(batch_limit)
    for i, batched_ics_ in enumerate(batched_ics):
        # optimize using random restart optimization
        batch_candidates_curr, batch_acq_values_curr = gen_candidates_scipy(
            initial_conditions=batched_ics_,
            acquisition_function=acq_function,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
            options={
                k: v
                for k, v in options.items()
                if k not in ("init_batch_limit", "batch_limit", "nonnegative")
            },
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            precision = kwargs.get('precision')
        )
        batch_candidates_list.append(batch_candidates_curr)
        batch_acq_values_list.append(batch_acq_values_curr)
        logger.info(f"Generated candidate batch {i+1} of {len(batched_ics)}.")
    batch_candidates = torch.cat(batch_candidates_list)
    batch_acq_values = torch.cat(batch_acq_values_list)

    if post_processing_func is not None:
        batch_candidates = post_processing_func(batch_candidates)

    if return_best_only:
        best = torch.argmax(batch_acq_values.view(-1), dim=0)
        batch_candidates = batch_candidates[best]
        batch_acq_values = batch_acq_values[best]

    if isinstance(acq_function, OneShotAcquisitionFunction):
        if not kwargs.get("return_full_tree", False):
            batch_candidates = acq_function.extract_candidates(X_full=batch_candidates)

    return batch_candidates, batch_acq_values, X_rnd, Y_rnd


def gen_batch_initial_conditions2(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    additional_samples: Optional[Tensor] = None
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimziation.

    TODO: Support t-batches of initial conditions.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.
    """
    options = options or {}
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    batch_initial_arms: Tensor
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    bounds = bounds.cpu()
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = botorch.optim.initializers.initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = botorch.optim.initializers.initialize_q_batch

    q = 1 if q is None else q

    while factor < max_factor:
        n = raw_samples * factor
        X_rnd = draw_sobol_samples(bounds=bounds, n=n, q=q, seed=seed)
        if additional_samples is not None:
            X_rnd = torch.cat((additional_samples, X_rnd), dim=0)
        X_rnd = fix_features(X_rnd, fixed_features=fixed_features)
        print(X_rnd.shape)
        with torch.no_grad():
            if batch_limit is None:
                batch_limit = X_rnd.shape[0]
            Y_rnd_list = []
            start_idx = 0
            while start_idx < X_rnd.shape[0]:
                end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                Y_rnd_curr = acq_function(
                    X_rnd[start_idx:end_idx].to(device=device)
                ).cpu()
                print(X_rnd[start_idx:end_idx].shape)
                print(Y_rnd_curr.shape)
                asdasdasd

                Y_rnd_list.append(Y_rnd_curr)
                start_idx += batch_limit
            Y_rnd = torch.cat(Y_rnd_list)
        # print(Y_rnd.argsort(descending=True)[:10])
        # print(Y_rnd[Y_rnd.argsort(descending=True)[:10]])
        # print(X_rnd[Y_rnd.argsort(descending=True)[:10]])
        # print(Y_rnd[:10])
        # print(X_rnd[:10])
        batch_initial_conditions = init_func(
            X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
        ).to(device=device)
        if factor < max_factor:
            factor += 1
            if seed is not None:
                seed += 1  # make sure to sample different X_rnd
    return batch_initial_conditions, X_rnd, Y_rnd