import warnings
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression


def predict_ar_single(
    series: NDArray[np.float64],
    p: int,
) -> tuple[float, NDArray[np.float64]]:
    """
    Calculate 1-step ahead AR(p) forecast for a single return series.

    Parameters
    ----------
    series : NDArray[np.float64]
        A 1D array of shape (T,) representing return observations.
    p : int
        The lag order for the autoregressive model.

    Returns
    -------
    forecast : float
        The 1-step ahead forecast value.
    residuals : NDArray[np.float64]
        A 1D array of shape (T - p,) containing in-sample residuals.

    Raises
    ------
    ValueError
        If series is not 1D, has zero dimension, or if p is outside valid range.
    """

    if series.ndim != 1:
        raise ValueError(f"series must be 1D, got {series.ndim}D")
    t_obs = series.shape[0]
    if t_obs == 0:
        raise ValueError("series must have non-zero dimension")
    if p <= 0:
        raise ValueError(f"p must be a positive integer, got {p}")
    if p >= t_obs:
        raise ValueError(f"p ({p}) must be less than sample size ({t_obs})")

    X = np.column_stack([series[p - i - 1 : t_obs - i - 1] for i in range(p)])
    y = series[p:]

    model = LinearRegression()
    model.fit(X, y)

    fitted_y = model.predict(X)
    residuals = y - fitted_y

    x_pred = series[-p:][::-1].reshape(1, -1)
    forecast = float(model.predict(x_pred)[0])

    return forecast, residuals


def _predict_naive_mean(
    returns_matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead forecast using historical column means.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape (T, N) where rows represent time periods and
        columns represent asset returns.

    Returns
    -------
    forecasts : NDArray[np.float64]
        A 1D array of shape (N,) containing the historical mean forecast.
    residuals : NDArray[np.float64]
        A 2D array of shape (T, N) containing in-sample residuals.

    Raises
    ------
    ValueError
        If returns_matrix is not 2D or has zero dimensions.
    """

    if returns_matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got {returns_matrix.ndim}D")
    if returns_matrix.shape[0] == 0 or returns_matrix.shape[1] == 0:
        raise ValueError("returns_matrix must have non-zero dimensions")

    forecasts = cast(NDArray[np.float64], np.mean(returns_matrix, axis=0))
    residuals = returns_matrix - forecasts

    return forecasts, residuals


def _predict_ar_matrix(
    returns_matrix: NDArray[np.float64],
    p_list: list[int] | int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead AR forecasts for each asset in a returns matrix.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape (T, N) where rows represent time periods and
        columns represent asset returns.
    p_list : list[int] | int, default=1
        Lag order for each series, or a single integer applied to all series.

    Returns
    -------
    forecasts : NDArray[np.float64]
        A 1D array of shape (N,) containing 1-step ahead AR forecasts.
    residuals : NDArray[np.float64]
        A 2D array of shape (T - p_max, N) containing aligned residuals.

    Raises
    ------
    ValueError
        If returns_matrix is invalid or p_list length does not match asset count.
    """

    if returns_matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got {returns_matrix.ndim}D")
    t_obs, num_assets = returns_matrix.shape
    if t_obs == 0 or num_assets == 0:
        raise ValueError("returns_matrix must have non-zero dimensions")

    if isinstance(p_list, int):
        p_orders = [p_list] * num_assets
    else:
        p_orders = list(p_list)

    if len(p_orders) != num_assets:
        raise ValueError(f"p_list length ({len(p_orders)}) must match number of assets ({num_assets})")

    p_max = max(p_orders)
    if p_max >= t_obs:
        raise ValueError(f"maximum p ({p_max}) must be less than sample size ({t_obs})")

    forecasts = np.zeros(num_assets, dtype=np.float64)
    residuals_list = []

    for col_idx in range(num_assets):
        f_val, res = predict_ar_single(returns_matrix[:, col_idx], p_orders[col_idx])
        forecasts[col_idx] = f_val
        # Align time dimensions across assets by trimming to the last (T - p_max) observations
        residuals_list.append(res[-(t_obs - p_max) :])

    residuals = np.column_stack(residuals_list)

    return forecasts, residuals


def _predict_var_lasso(
    returns_matrix: NDArray[np.float64],
    p: int = 1,
    alpha: float = 1.0,
    tol: float = 1e-4,
    max_iter: int = 10000,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead VAR(p) forecast with Lasso regularization.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape (T, N) where rows represent time periods and
        columns represent asset returns.
    p : int, default=1
        The lag order for the VAR model.
    alpha : float, default=1.0
        Lasso regularization strength parameter.
    tol : float, default=1e-4
        Tolerance for optimization convergence.
    max_iter : int, default=10000
        Maximum number of iterations for coordinate descent.

    Returns
    -------
    forecasts : NDArray[np.float64]
        A 1D array of shape (N,) containing 1-step ahead VAR-Lasso forecasts.
    residuals : NDArray[np.float64]
        A 2D array of shape (T - p, N) containing in-sample residuals.

    Raises
    ------
    ValueError
        If returns_matrix is invalid or parameters fall outside valid ranges.
    """

    if returns_matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got {returns_matrix.ndim}D")
    t_obs, num_assets = returns_matrix.shape
    if t_obs == 0 or num_assets == 0:
        raise ValueError("returns_matrix must have non-zero dimensions")
    if p <= 0:
        raise ValueError(f"p must be a positive integer, got {p}")
    if p >= t_obs:
        raise ValueError(f"p ({p}) must be less than sample size ({t_obs})")
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    if tol <= 0.0:
        raise ValueError(f"tol must be positive, got {tol}")
    if max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer, got {max_iter}")

    X = np.hstack([returns_matrix[p - i - 1 : t_obs - i - 1] for i in range(p)])
    Y = returns_matrix[p:]

    if alpha == 0.0:
        model: Lasso | LinearRegression = LinearRegression()
        model.fit(X, Y)
    else:
        model = Lasso(alpha=alpha, tol=tol, max_iter=max_iter)
        # Fail fast if Lasso doesn't converge
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            try:
                model.fit(X, Y)
            except ConvergenceWarning as e:
                raise ValueError(f"VAR Lasso failed to converge within {max_iter} iterations") from e

    fitted_Y = np.asarray(model.predict(X)).reshape(Y.shape)
    residuals = Y - fitted_Y

    x_pred = returns_matrix[-p:][::-1].reshape(1, -1)
    pred = model.predict(x_pred)
    forecasts = np.asarray(pred, dtype=np.float64).ravel()

    return forecasts, residuals


def predict_mean(
    returns_matrix: NDArray[np.float64],
    model: Literal["naive", "ar", "var"] = "naive",
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead return forecast using the specified mean model.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape (T, N) where rows represent time periods and
        columns represent asset returns.
    model : Literal["naive", "ar", "var"], default="naive"
        The forecasting model to use.
    **kwargs : Any
        Keyword arguments passed directly to the underlying model implementation
        (e.g., `p_list`, `p`, `alpha`).

    Returns
    -------
    forecasts : NDArray[np.float64]
        A 1D array of shape (N,) containing 1-step ahead return forecasts.
    residuals : NDArray[np.float64]
        A 2D array of shape (T_eff, N) containing in-sample residuals.

    Raises
    ------
    ValueError
        If model is unrecognized or arguments are invalid.
    """

    if model == "naive":
        return _predict_naive_mean(returns_matrix)
    if model == "ar":
        return _predict_ar_matrix(returns_matrix, **kwargs)
    if model == "var":
        return _predict_var_lasso(returns_matrix, **kwargs)

    raise ValueError(f"unknown model '{model}', supported models are 'naive', 'ar', 'var'")
