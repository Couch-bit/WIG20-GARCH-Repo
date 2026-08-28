import random
from collections.abc import Callable
from typing import cast

import numpy as np
from deap import base, creator, tools
from numpy.typing import NDArray

from src.config import TOTAL_TOKENS


def optimize_portfolio(
    returns_matrix: NDArray[np.float64],
    metric_func: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
    ngen: int = 40,
    pop_size: int = 100,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    max_shift: int = 20,
    indpb: float = 0.5,
    tournsize: int = 2,
    total_tokens: int = TOTAL_TOKENS,
    warm_start_pop: list[list[int]] | None = None,
    warm_start_ratio: float = 0.2,
) -> tuple[NDArray[np.float64], list[list[int]]]:
    """
    Optimize portfolio allocation using a Genetic Algorithm with pairwise comparison.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array where rows represent samples/time periods and columns represent
        assets, containing return values.
    metric_func : Callable[[NDArray[np.float64], NDArray[np.float64]], float]
        A evaluation callable accepting `returns_matrix` (2D) and a 1D numpy array of
        portfolio weights, returning a numeric metric float.
    ngen : int, default=40
        Number of evolution generations.
    pop_size : int, default=100
        Number of individuals in the population.
    cxpb : float, default=0.7
        Probability of crossover between parents.
    mutpb : float, default=0.2
        Probability of individual mutation.
    max_shift: int, default=20
        The maximum possible shift of tokens during mutation.
    indpb : float, default=0.5
        Independent probability for each attribute to be exchanged during crossover.
    tournsize : int, default=2
        Number of candidates during each round of tournament selection.
    total_tokens : int, default=TOTAL_TOKENS
        Total integer tokens for allocation resolution (step size = 1/total_tokens).
    warm_start_pop : list[list[int]] | None, default=None
        Optional population of token chromosomes (must have length equal to pop_size) to seed the initial generation.
    warm_start_ratio : float, default=0.2
        Proportion of the initial population to fill with best warm-start individuals.

    Returns
    -------
    best_weights : NDArray[np.float64]
        A 1D array of portfolio weights (summing to 1.0) for the optimal individual.
    final_population : list[list[int]]
        A list of token chromosomes representing the final population at the end of evolution.

    Raises
    ------
    ValueError
        If `returns_matrix` is not 2D, contains zero dimensions, or parameters fall
        outside valid mathematical ranges.
    """

    # Validate values
    if returns_matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got {returns_matrix.ndim}D")
    if returns_matrix.shape[0] == 0 or returns_matrix.shape[1] == 0:
        raise ValueError("returns_matrix must have non-zero dimensions")
    if returns_matrix.shape[1] < 2:
        raise ValueError(f"returns_matrix must have at least 2 assets, got {returns_matrix.shape[1]}")
    if ngen <= 0:
        raise ValueError(f"ngen must be a positive integer, got {ngen}")
    if pop_size <= 0:
        raise ValueError(f"pop_size must be a positive integer, got {pop_size}")
    if pop_size % 2 != 0:
        raise ValueError(f"pop_size must be an even integer, got {pop_size}")
    if not (0.0 <= cxpb <= 1.0):
        raise ValueError(f"cxpb must be between 0.0 and 1.0, got {cxpb}")
    if not (0.0 <= mutpb <= 1.0):
        raise ValueError(f"mutpb must be between 0.0 and 1.0, got {mutpb}")
    if max_shift <= 0:
        raise ValueError(f"max_shift must be a positive integer, got {max_shift}")
    if not (0.0 <= indpb <= 1.0):
        raise ValueError(f"indpb must be between 0.0 and 1.0, got {indpb}")
    if not (2 <= tournsize <= pop_size):
        raise ValueError(f"tournsize must be between 2 and pop_size ({pop_size}), got {tournsize}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be a positive integer, got {total_tokens}")
    if not (0.0 <= warm_start_ratio <= 1.0):
        raise ValueError(f"warm_start_ratio must be between 0.0 and 1.0, got {warm_start_ratio}")

    num_assets = returns_matrix.shape[1]

    if warm_start_pop is not None:
        if len(warm_start_pop) != pop_size:
            raise ValueError(
                f"warm_start_pop must have length equal to pop_size ({pop_size}), got {len(warm_start_pop)}"
            )
        for ind in warm_start_pop:
            if len(ind) != num_assets:
                raise ValueError(f"Each individual in warm_start_pop must have length {num_assets}")
            if sum(ind) != total_tokens:
                raise ValueError(f"Each individual in warm_start_pop must sum to total_tokens ({total_tokens})")
            if any(x < 0 for x in ind):
                raise ValueError("Token counts in warm_start_pop must be non-negative integers")

    asset_means = np.mean(returns_matrix, axis=0)

    # DEAP Class Creation
    if not hasattr(creator, "FitnessPlaceholder"):
        creator.create("FitnessPlaceholder", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessPlaceholder)

    # Define helpers
    def create_individual(k_assets: int) -> list[int]:
        cuts = sorted([random.randint(0, total_tokens) for _ in range(k_assets - 1)])
        cuts = [0] + cuts + [total_tokens]
        return cast(list[int], creator.Individual([cuts[i] - cuts[i - 1] for i in range(1, len(cuts))]))

    def mut_zero_sum(individual: list[int]) -> tuple[list[int]]:
        positive_indices = [i for i, val in enumerate(individual) if val > 0]
        idx_a = random.choice(positive_indices)
        other_indices = [i for i in range(len(individual)) if i != idx_a]
        idx_b = random.choice(other_indices)

        shift = random.randint(1, min(individual[idx_a], max_shift))
        individual[idx_a] -= shift
        individual[idx_b] += shift

        return (individual,)

    def cx_uniform_repair(ind1: list[int], ind2: list[int]) -> tuple[list[int], list[int]]:
        tools.cxUniform(ind1, ind2, indpb)

        # Repair
        for ind in (ind1, ind2):
            current_sum = sum(ind)

            # Edge case: if crossover somehow yielded all zeros
            if current_sum == 0:
                ind[:] = create_individual(len(ind))
                continue

            # Compute exact proportional targets
            exact = [x * total_tokens / current_sum for x in ind]

            # Base allocation via flooring (guarantees sum <= total_tokens and all values >= 0)
            base_alloc = [int(f) for f in exact]
            remainder = total_tokens - sum(base_alloc)  # Always 0 <= remainder < len(ind)

            # Sort fractional remainders descending
            fractional = [(exact[i] - base_alloc[i], i) for i in range(len(ind))]
            fractional.sort(key=lambda item: item[0], reverse=True)

            # Distribute 1 token to the top 'remainder' largest fractional parts
            for k in range(remainder):
                idx = fractional[k][1]
                base_alloc[idx] += 1

            ind[:] = base_alloc

        return ind1, ind2

    def safe_evaluate(ind: list[int]) -> tuple[float | None, bool]:
        weights = np.array(ind, dtype=float) / total_tokens
        try:
            score = metric_func(returns_matrix, weights)
            return score, False
        except ValueError:
            return None, True

    def is_better(ind1: list[int], ind2: list[int]) -> bool:
        score1, err1 = safe_evaluate(ind1)
        score2, err2 = safe_evaluate(ind2)

        def get_mean_return(ind: list[int]) -> float:
            w = np.array(ind, dtype=float) / total_tokens
            return float(np.dot(asset_means, w))

        # If ind1 errors and ind2 does not: ind1 is better if mean_return >= 0
        if err1 and not err2:
            return get_mean_return(ind1) >= 0.0

        # If ind2 errors and ind1 does not: ind2 is better if mean_return >= 0 (so ind1 is worse)
        if not err1 and err2:
            return get_mean_return(ind2) < 0.0

        # If both error out, pick the one with higher mean return
        if err1 and err2:
            return get_mean_return(ind1) > get_mean_return(ind2)

        # If neither errors out, directly compare metrics
        return cast(float, score1) > cast(float, score2)

    def sel_custom_tournament(individuals: list[list[int]], k: int) -> list[list[int]]:
        chosen = []
        for _ in range(k):
            aspirants = random.sample(individuals, tournsize)
            winner = aspirants[0]
            for competitor in aspirants[1:]:
                if is_better(competitor, winner):
                    winner = competitor
            chosen.append(winner)
        return chosen

    # DEAP Toolbox Registration
    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual, num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", cx_uniform_repair)
    toolbox.register("mutate", mut_zero_sum)
    toolbox.register("select", sel_custom_tournament)

    # Initial Population Generation (with Warm-Start ranking)
    if warm_start_pop is not None and warm_start_ratio > 0.0:
        num_warm = min(int(round(pop_size * warm_start_ratio)), pop_size)
        if num_warm > 0:

            def warm_sort_key(ind: list[int]) -> tuple[int, float]:
                score, err = safe_evaluate(ind)
                w = np.array(ind, dtype=float) / total_tokens
                mean_ret = float(np.dot(asset_means, w))

                if err:
                    if mean_ret >= 0.0:
                        return (2, mean_ret)
                    return (0, mean_ret)

                return (1, cast(float, score))

            # Rank the warm-start population and pick the top `num_warm` best individuals
            sorted_warm = sorted(warm_start_pop, key=warm_sort_key, reverse=True)
            best_warm = sorted_warm[:num_warm]

            warm_individuals = [cast(list[int], creator.Individual(list(ind))) for ind in best_warm]
            random_individuals = [toolbox.individual() for _ in range(pop_size - num_warm)]
            pop = warm_individuals + random_individuals
        else:
            pop = toolbox.population(n=pop_size)
    else:
        pop = toolbox.population(n=pop_size)

    # GA Optimization Loop with 1-Elitism
    for _ in range(ngen):
        # Track the Elite individual
        elite = pop[0]
        for ind in pop[1:]:
            if is_better(ind, elite):
                elite = ind
        elite = toolbox.clone(elite)

        # Tournament Selection
        offspring = toolbox.select(pop, k=len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2], strict=True):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)

        # Mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)

        # Enforce Elitism
        offspring[0] = elite
        pop = offspring

    # Extract overall best individual
    best_ind = pop[0]
    for ind in pop[1:]:
        if is_better(ind, best_ind):
            best_ind = ind

    best_weights = np.array(best_ind, dtype=float) / total_tokens
    final_population = [list(ind) for ind in pop]

    return best_weights, final_population
