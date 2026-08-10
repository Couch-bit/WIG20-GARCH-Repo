from typing import cast

import numpy as np
import scipy.linalg
from numpy.typing import NDArray


def multivariate_normal_log_likelihood(
    x: NDArray[np.float64],
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """
    Calculate the log-likelihood of sample(s) under a multivariate normal distribution.

    Uses Cholesky decomposition for optimal numerical stability and efficiency,
    avoiding direct matrix inversion and log-determinant underflow/overflow.

    Parameters
    ----------
    x : NDArray[np.float64]
        Input observation vector of shape (d,) or batch of vectors of shape (N, d),
        where 'd' is the feature dimension and 'N' is the number of samples.
    mean : NDArray[np.float64]
        The mean vector of the distribution of shape (d,).
    cov : NDArray[np.float64]
        The covariance matrix of the distribution of shape (d, d).

    Returns
    -------
    float | NDArray[np.float64]
        The log-likelihood value as a float (if 'x' is a single 1D vector) or a 1D
        NumPy array of log-likelihood values of shape (N,) (if 'x' is a 2D batch).

    Raises
    ------
    ValueError
        If dimensions or shapes between inputs are mismatched.
    scipy.linalg.LinAlgError
        If Cholesky decomposition fails (e.g., if the covariance matrix is not positive-definite).
    """

    # Shape Validation Logic
    if mean.ndim != 1:
        raise ValueError(f"Expected 'mean' to be a 1D array, got a {mean.ndim}D array")

    if cov.ndim != 2:
        raise ValueError(f"Expected 'cov' to be a 2D matrix, got a {cov.ndim}D matrix")

    d = cast(int, mean.shape[0])

    if cov.shape != (d, d):
        raise ValueError(f"Covariance matrix shape {cov.shape} does not match mean vector length ({d}, {d})")

    # Normalize 'x' to a 2D batch array internally while validating dimensions
    if x.ndim == 1:
        if x.shape[0] != d:
            raise ValueError(f"Feature dimension of 'x' ({x.shape[0]}) does not match mean dimension ({d})")
        is_single_sample = True
        x_batch = x.reshape(1, d)
    elif x.ndim == 2:
        if x.shape[1] != d:
            raise ValueError(f"Feature dimension of 'x' ({x.shape[1]}) does not match mean dimension ({d})")
        is_single_sample = False
        x_batch = x
    else:
        raise ValueError(f"Expected 'x' to be a 1D vector or 2D batch array, got a {x.ndim}D array")

    # Computation using Cholesky Decomposition
    try:
        # L is lower triangular such that Cov = L @ L.T
        L = scipy.linalg.cholesky(cov, lower=True)
    except scipy.linalg.LinAlgError as err:
        raise scipy.linalg.LinAlgError(
            "Cholesky decomposition failed. Ensure the covariance matrix is positive-definite"
        ) from err

    # Log-determinant term: log|Cov| = 2 * sum(log(diag(L)))
    log_det_cov = cast(float, 2.0 * np.sum(np.log(np.diag(L))))

    # Squared Mahalanobis distance term: (x - mu).T @ Cov^-1 @ (x - mu)
    diff = (x_batch - mean).T
    z = scipy.linalg.solve_triangular(L, diff, lower=True)
    mahalanobis_sq = cast(NDArray[np.float64], np.sum(z**2, axis=0))

    # Log-likelihood evaluation formula
    log_2pi = cast(float, np.log(2.0 * np.pi))
    log_likelihood = -0.5 * (d * log_2pi + log_det_cov + mahalanobis_sq)

    if is_single_sample:
        return float(log_likelihood[0])

    return log_likelihood


def validate_array_and_bounds(returns: NDArray[np.float64], alpha: float | None = None) -> None:
    """
    Validate returns array dimensionality, non-emptiness, and tail probability bounds.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    alpha : float | None, default=None
        Tail probability parameter for Expected Shortfall. If provided, must be
        strictly between 0.0 and 1.0 (exclusive).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If 'returns' is not a 1D array, is empty, or if 'alpha' is out of bounds (0, 1).
    """

    if returns.ndim != 1:
        raise ValueError(f"Expected 'returns' to be a 1D array, got a {returns.ndim}D array")

    if returns.size == 0:
        raise ValueError("'returns' array cannot be empty.")

    if alpha is not None and not (0.0 < alpha < 1.0):
        raise ValueError(f"'alpha' must be strictly between 0 and 1 (exclusive), got {alpha}")


def expected_shortfall(excess_returns: NDArray[np.float64], alpha: float = 0.05) -> float:
    """
    Calculate sample Expected Shortfall (CVaR) for excess returns at tail probability alpha.

    Parameters
    ----------
    excess_returns : NDArray[np.float64]
        1D array of excess returns.
    alpha : float, default=0.05
        Tail probability parameter.

    Returns
    -------
    float
        The estimated Expected Shortfall.

    Raises
    ------
    ValueError
        If 'returns' is not a 1D array, is empty, or if 'alpha' is out of bounds (0, 1).
    """

    validate_array_and_bounds(excess_returns, alpha)
    var_threshold = float(np.quantile(excess_returns, alpha))
    tail_returns = excess_returns[excess_returns <= var_threshold]
    return float(-np.mean(tail_returns))


def sharpe_ratio(
    returns: NDArray[np.float64],
    rf: float = 0.0,
    ddof: int = 1,
) -> float:
    """
    Calculate the Sharpe ratio of a sample of returns.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    rf : float, default=0.0
        Risk-free rate of return.
    ddof : int, default=1
        Delta degrees of freedom used in standard deviation calculation.

    Returns
    -------
    float
        The calculated Sharpe ratio.

    Raises
    ------
    ValueError
        If returns array is invalid or standard deviation is zero.
    """

    validate_array_and_bounds(returns)

    excess_returns = returns - rf
    expected_excess_return = float(np.mean(excess_returns))
    volatility = float(np.std(excess_returns, ddof=ddof))

    if volatility == 0.0:
        raise ValueError("Standard deviation of excess returns is zero; Sharpe ratio is undefined")

    return expected_excess_return / volatility


def central_sortino_ratio(
    returns: NDArray[np.float64],
    rf: float = 0.0,
) -> float:
    """
    Calculate the Central Sortino ratio of a sample of returns.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    rf : float, default=0.0
        Risk-free rate of return.

    Returns
    -------
    float
        The calculated Central Sortino ratio.

    Raises
    ------
    ValueError
        If returns array is invalid or central downside deviation is zero.
    """

    validate_array_and_bounds(returns)

    expected_return = float(np.mean(returns))
    expected_excess_return = expected_return - float(rf)

    # Downside difference relative to mean return: (E[R] - R)_+
    downside_diff = np.maximum(expected_return - returns, 0.0)
    central_downside_dev = float(np.sqrt(np.mean(downside_diff**2)))

    if central_downside_dev == 0.0:
        raise ValueError("Central downside deviation is zero; Central Sortino ratio is undefined")

    return expected_excess_return / central_downside_dev


def central_tail_effectiveness_ratio(
    returns: NDArray[np.float64],
    rf: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """
    Calculate the Central Tail Effectiveness ratio of a sample of returns.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    rf : float, default=0.0
        Risk-free rate of return.
    alpha : float, default=0.05
        Tail probability level for Expected Shortfall.

    Returns
    -------
    float
        The calculated Central Tail Effectiveness ratio.

    Raises
    ------
    ValueError
        If inputs are invalid or the denominator is zero.
    """

    validate_array_and_bounds(returns, alpha)

    excess_returns = returns - rf
    expected_excess_return = float(np.mean(excess_returns))
    es = expected_shortfall(excess_returns, alpha)

    denominator = es + expected_excess_return

    if denominator == 0.0:
        raise ValueError("Denominator (ES_alpha + E[R - Rf]) is zero; ratio is undefined")

    return expected_excess_return / denominator


def omega_ratio(
    returns: NDArray[np.float64],
    rf: float = 0.0,
) -> float:
    """
    Calculate the Omega ratio of a sample of returns.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    rf : float, default=0.0
        Risk-free rate of return used as the threshold benchmark.

    Returns
    -------
    float
        The calculated Omega ratio.

    Raises
    ------
    ValueError
        If returns array is invalid or expected downside is zero.
    """

    validate_array_and_bounds(returns)

    excess_returns = returns - rf
    expected_excess_return = float(np.mean(excess_returns))

    # Downside relative to risk-free rate threshold: (Rf - R)_+
    downside = np.maximum(rf - returns, 0.0)
    expected_downside = float(np.mean(downside))

    if expected_downside == 0.0:
        raise ValueError("Expected downside below risk-free rate is zero; Omega ratio is undefined")

    return expected_excess_return / expected_downside


def sortino_ratio(
    returns: NDArray[np.float64],
    rf: float = 0.0,
) -> float:
    """
    Calculate the standard Sortino ratio of a sample of returns.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    rf : float, default=0.0
        Risk-free rate of return used as the threshold benchmark.

    Returns
    -------
    float
        The calculated Sortino ratio.

    Raises
    ------
    ValueError
        If returns array is invalid or downside deviation is zero.
    """

    validate_array_and_bounds(returns)

    excess_returns = returns - rf
    expected_excess_return = float(np.mean(excess_returns))

    # Downside relative to risk-free rate threshold: (Rf - R)_+
    downside = np.maximum(rf - returns, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)))

    if downside_dev == 0.0:
        raise ValueError("Downside deviation below risk-free rate is zero; Sortino ratio is undefined")

    return expected_excess_return / downside_dev


def tail_effectiveness_ratio(
    returns: NDArray[np.float64],
    rf: float = 0.0,
    alpha: float = 0.05,
) -> float:
    """
    Calculate the Tail Effectiveness ratio of a sample of returns.

    Parameters
    ----------
    returns : NDArray[np.float64]
        1D array of sample asset returns.
    rf : float, default=0.0
        Risk-free rate of return $R_f$.
    alpha : float, default=0.05
        Tail probability level for Expected Shortfall $\\alpha \\in (0, 1)$.

    Returns
    -------
    float
        The calculated Tail Effectiveness ratio $TE^f_\\alpha(R)$.

    Raises
    ------
    ValueError
        If inputs are invalid or Expected Shortfall is zero.
    """

    validate_array_and_bounds(returns, alpha)

    excess_returns = returns - rf
    expected_excess_return = float(np.mean(excess_returns))
    es = expected_shortfall(excess_returns, alpha)

    if es <= 0.0:
        raise ValueError("Expected Shortfall is non-positive; Tail Effectiveness ratio is undefined")

    return expected_excess_return / es
