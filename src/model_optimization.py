import warnings
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from src.config import WINDOW_SIZE
from src.metrics import multivariate_normal_log_likelihood, univariate_normal_log_likelihood
from src.models import (
    MeanModel,
    UGARCHModel,
    VolatilityModel,
    predict_ar_single,
    predict_mean,
    predict_univariate_garch,
    predict_volatility,
)

#############
### Setup ###
#############
_DEFAULT_AR_P_GRID = list(range(21))
_DEFAULT_VAR_P_GRID = list(range(1, 6))
_DEFAULT_VAR_LASSO_P_GRID = list(range(1, 21))
_DEFAULT_ALPHA_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1]

_DEFAULT_UGARCH_GRID: list[UGARCHModel] = ["sGARCH", "eGARCH", "gjrGARCH"]
_DEFAULT_ASYMMETRIC_GRID = [False, True]


######################
### Tuning Helpers ###
######################
def _validate_returns_matrix(
    returns_matrix: NDArray[np.float64],
    window_size: int,
) -> tuple[int, int]:
    """
    Validate returns matrix dimensions and rolling window size constraints.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` containing asset return observations.
    window_size : int
        Number of historical observations in each rolling evaluation window.

    Returns
    -------
    tuple[int, int]
        Sample size `T` and asset count `N`.

    Raises
    ------
    ValueError
        If inputs do not satisfy shape or window size constraints.
    """

    if returns_matrix.ndim != 2:
        raise ValueError(f"Expected 2D returns_matrix, got a {returns_matrix.ndim}D array")

    t_obs, num_assets = returns_matrix.shape
    if t_obs == 0 or num_assets == 0:
        raise ValueError("returns_matrix must have non-zero dimensions")

    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    if window_size >= t_obs:
        raise ValueError(f"window_size ({window_size}) must be strictly less than sample size ({t_obs})")

    return t_obs, num_assets


def _tune_ar_matrix(
    returns_matrix: NDArray[np.float64],
    window_size: int,
    p_grid: list[int] | None = None,
    var_eps: float = 1e-12,
) -> dict[str, Any]:
    """
    Tune AR(p) lag order independently for each asset series.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` containing asset return observations.
    window_size : int
        Number of historical observations in each rolling window.
    p_grid : list[int] | None, default=None
        Candidate AR lag orders to evaluate (defaults to `1..20`).
    var_eps : float, default=1e-12
        Minimum variance threshold to prevent numerical instability.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `'p_list'`: list of optimal lag orders per asset.

    Raises
    ------
    ValueError
        If inputs do not satisfy shape/window constraints, if `p_grid` contains no
        valid lags, or if no candidate lag order converges for an asset.
    """

    t_obs, num_assets = _validate_returns_matrix(returns_matrix, window_size)
    lags_to_test = list(p_grid) if p_grid else _DEFAULT_AR_P_GRID

    # Filter out lag orders larger than window_size - 1
    valid_lags = [p for p in lags_to_test if 0 <= p < window_size]
    if not valid_lags:
        raise ValueError("No valid lag orders in p_grid relative to window_size")

    best_p_list: list[int] = []

    for col_idx in range(num_assets):
        series = returns_matrix[:, col_idx]
        best_p = None
        best_total_ll = -np.inf

        for p in valid_lags:
            total_ll = 0.0
            valid_candidate = True

            for start_idx in range(t_obs - window_size):
                window_data = series[start_idx : start_idx + window_size]
                test_obs = series[start_idx + window_size]

                try:
                    fcst, res = predict_ar_single(window_data, p=p)
                    var_in = cast(float, np.mean(res**2))
                    if var_in <= var_eps:
                        valid_candidate = False
                        break
                    ll = univariate_normal_log_likelihood(test_obs, mean=fcst, variance=var_in)
                    total_ll += float(ll)
                except Exception:
                    valid_candidate = False
                    break

            if valid_candidate and total_ll > best_total_ll:
                best_total_ll = total_ll
                best_p = p

        if best_p is None or best_total_ll == -np.inf:
            raise ValueError(f"No valid lag order converged for asset index {col_idx}")

        best_p_list.append(best_p)

    return {"p_list": best_p_list}


def _tune_var(
    returns_matrix: NDArray[np.float64],
    window_size: int,
    p_grid: list[int] | None = None,
    cov_eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Tune VAR(p) model lag order across all asset series jointly.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` containing asset return observations.
    window_size : int
        Number of historical observations in each rolling window.
    p_grid : list[int] | None, default=None
        Candidate VAR lag orders to evaluate (defaults to `1..5`).
    cov_eps : float, default=1e-8
        Diagonal regularization constant added to in-sample covariance matrices.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `'p'`: optimal VAR lag order.

    Raises
    ------
    ValueError
        If inputs do not satisfy shape/window constraints, if `p_grid` contains no
        valid lags, or if no candidate lag order converges.
    """

    t_obs, num_assets = _validate_returns_matrix(returns_matrix, window_size)
    lags_to_test = list(p_grid) if p_grid else _DEFAULT_VAR_P_GRID

    valid_lags = [p for p in lags_to_test if 0 < p < window_size]
    if not valid_lags:
        raise ValueError("No valid lag orders in p_grid relative to window_size")

    best_p = None
    best_total_ll = -np.inf

    for p in valid_lags:
        total_ll = 0.0
        valid_candidate = True

        for start_idx in range(t_obs - window_size):
            window_data = returns_matrix[start_idx : start_idx + window_size]
            test_obs = returns_matrix[start_idx + window_size]

            try:
                fcst, res = predict_mean(window_data, "var", p=p, alpha=0.0)
                n_samples = res.shape[0]
                cov_in = (res.T @ res) / n_samples
                cov_in += np.eye(num_assets) * cov_eps

                ll = multivariate_normal_log_likelihood(test_obs, mean=fcst, cov=cov_in)
                total_ll += float(ll)
            except Exception:
                valid_candidate = False
                break

        if valid_candidate and total_ll > best_total_ll:
            best_total_ll = total_ll
            best_p = p

    if best_p is None or best_total_ll == -np.inf:
        raise ValueError("No valid VAR lag order converged across candidate grid")

    return {"p": best_p}


def _tune_var_lasso(
    returns_matrix: NDArray[np.float64],
    window_size: int,
    p_grid: list[int] | None = None,
    alpha_grid: list[float] | None = None,
    cov_eps: float = 1e-8,
) -> dict[str, Any]:
    """
    Tune VAR(p) model with Lasso regularization over lag orders and L1 penalties.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` containing asset return observations.
    window_size : int
        Number of historical observations in each rolling window.
    p_grid : list[int] | None, default=None
        Candidate VAR lag orders to evaluate (defaults to `1..20`).
    alpha_grid : list[float] | None, default=None
        Candidate Lasso L1 regularization parameters to evaluate.
    cov_eps : float, default=1e-8
        Diagonal regularization constant added to in-sample covariance matrices.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `'p'`: optimal lag order, and `'alpha'`: optimal L1 penalty.

    Raises
    ------
    ValueError
        If inputs do not satisfy shape/window constraints, if `p_grid` contains no
        valid lags, if there are no valid alphas, or if no candidate (p, alpha) combination converges.
    """

    t_obs, num_assets = _validate_returns_matrix(returns_matrix, window_size)

    lags_to_test = list(p_grid) if p_grid else _DEFAULT_VAR_LASSO_P_GRID
    raw_alphas = list(alpha_grid) if alpha_grid else _DEFAULT_ALPHA_GRID

    valid_lags = [p for p in lags_to_test if 0 < p < window_size]
    if not valid_lags:
        raise ValueError("No valid lag orders in p_grid relative to window_size")

    alphas_to_test = [a for a in raw_alphas if a >= 0.0]
    if not alphas_to_test:
        raise ValueError("No valid non-negative alpha values in alpha_grid")

    best_p = None
    best_alpha = None
    best_total_ll = -np.inf

    for p in valid_lags:
        for alpha in alphas_to_test:
            total_ll = 0.0
            valid_candidate = True

            for start_idx in range(t_obs - window_size):
                window_data = returns_matrix[start_idx : start_idx + window_size]
                test_obs = returns_matrix[start_idx + window_size]

                try:
                    fcst, res = predict_mean(window_data, "var", p=p, alpha=alpha)
                    n_samples = res.shape[0]
                    cov_in = (res.T @ res) / n_samples
                    cov_in += np.eye(num_assets) * cov_eps

                    ll = multivariate_normal_log_likelihood(test_obs, mean=fcst, cov=cov_in)
                    total_ll += float(ll)
                except Exception:
                    valid_candidate = False
                    break

            if valid_candidate and total_ll > best_total_ll:
                best_total_ll = total_ll
                best_p = p
                best_alpha = alpha

    if best_p is None or best_alpha is None or best_total_ll == -np.inf:
        raise ValueError("No valid VAR Lasso candidate (p, alpha) converged across candidate grid")

    return {"p": best_p, "alpha": best_alpha}


def _generate_rolling_residuals(
    returns_matrix: NDArray[np.float64],
    mean_model: MeanModel,
    window_size: int,
    mean_kwargs: dict[str, Any] | None = None,
) -> tuple[list[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Generate rolling in-sample mean residuals and out-of-sample test residuals.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` containing asset return observations.
    mean_model : MeanModel
        The mean forecasting model identifier.
    window_size : int
        Number of historical observations in each rolling evaluation window.
    mean_kwargs : dict[str, Any] | None, default=None
        Hyperparameters passed directly to `predict_mean`.

    Returns
    -------
    tuple[list[NDArray[np.float64]], NDArray[np.float64]]
        Tuple containing:

        * `window_residuals_list`: List of 2D arrays of shape `(T_eff, N)` containing in-sample mean residuals.
        * `test_residuals_matrix`: 2D array of shape `(K_valid, N)` containing out-of-sample mean test residuals.
    """

    t_obs, num_assets = returns_matrix.shape
    m_kwargs = mean_kwargs if mean_kwargs is not None else {}

    window_residuals_list: list[NDArray[np.float64]] = []
    test_residuals_list: list[NDArray[np.float64]] = []

    for start_idx in range(t_obs - window_size):
        window_data = returns_matrix[start_idx : start_idx + window_size]
        test_obs = returns_matrix[start_idx + window_size]

        try:
            fcst, in_sample_res = predict_mean(window_data, model=mean_model, **m_kwargs)
            test_res = test_obs - fcst
            window_residuals_list.append(in_sample_res)
            test_residuals_list.append(test_res)
        except Exception as e:
            warnings.warn(
                f"Mean model '{mean_model}' fitting failed at window index {start_idx}: {e}",
                category=UserWarning,
                stacklevel=2,
            )
            continue

    if not window_residuals_list:
        test_residuals_matrix = np.empty((0, num_assets), dtype=np.float64)
    else:
        test_residuals_matrix = np.vstack(test_residuals_list)

    return window_residuals_list, test_residuals_matrix


def _tune_univariate_garch_per_asset(
    window_residuals_list: list[NDArray[np.float64]],
    test_residuals_matrix: NDArray[np.float64],
    num_assets: int,
    u_grid: list[UGARCHModel] | None = None,
    var_eps: float = 1e-12,
) -> list[UGARCHModel]:
    """
    Tune univariate GARCH specification independently for each asset series (Stage 1).

    If no rolling window data exists or if all candidate models fail to converge, defaults to the
    first model in `u_grid`.

    Parameters
    ----------
    window_residuals_list : list[NDArray[np.float64]]
        List of 2D in-sample mean residual matrices for each rolling window.
    test_residuals_matrix : NDArray[np.float64]
        2D matrix of shape `(K_valid, N)` containing out-of-sample mean test residuals.
    num_assets : int
        Total number of asset series.
    u_grid : list[UGARCHModel] | None, default=None
        Candidate univariate GARCH specifications to evaluate (defaults to `'sGARCH'`, `'eGARCH'`, `'gjrGARCH'`).
    var_eps : float, default=1e-12
        Minimum variance threshold to prevent numerical instability.

    Returns
    -------
    list[UGARCHModel]
        List of optimal univariate GARCH model strings per asset series.

    Raises
    ------
    ValueError
        If `window_residuals_list` is empty or if no candidate univariate GARCH model
        converges for an asset series.
    """

    models_to_test = list(u_grid) if u_grid else _DEFAULT_UGARCH_GRID
    num_windows = len(window_residuals_list)

    if num_windows == 0:
        raise ValueError("No rolling window residual data available for tuning univariate GARCH models")

    best_u_models: list[UGARCHModel] = []
    zero_mean = 0.0

    for col_idx in range(num_assets):
        best_m = None
        best_total_ll = -np.inf

        for m in models_to_test:
            total_ll = 0.0
            valid_candidate = True

            for k in range(num_windows):
                series_window = window_residuals_list[k][:, col_idx]
                test_obs_res = test_residuals_matrix[k, col_idx]

                try:
                    var_f, _ = predict_univariate_garch(series_window, model=m)
                    if var_f <= var_eps:
                        valid_candidate = False
                        break
                    ll = univariate_normal_log_likelihood(test_obs_res, mean=zero_mean, variance=var_f)
                    total_ll += float(ll)
                except Exception:
                    valid_candidate = False
                    break

            if valid_candidate and total_ll > best_total_ll:
                best_total_ll = total_ll
                best_m = m

        if best_m is None or best_total_ll == -np.inf:
            raise ValueError(f"No valid univariate GARCH model converged for asset index {col_idx}")

        best_u_models.append(best_m)

    return best_u_models


def _tune_ccc(
    window_residuals_list: list[NDArray[np.float64]],
    test_residuals_matrix: NDArray[np.float64],
    num_assets: int,
    u_grid: list[UGARCHModel] | None = None,
) -> dict[str, Any]:
    """
    Tune Constant Conditional Correlation (CCC-GARCH) model via Stage 1 univariate GARCH selection.

    Parameters
    ----------
    window_residuals_list : list[NDArray[np.float64]]
        List of 2D in-sample mean residual matrices for each rolling window.
    test_residuals_matrix : NDArray[np.float64]
        2D matrix of shape `(K_valid, N)` containing out-of-sample mean test residuals.
    num_assets : int
        Total number of asset series.
    u_grid : list[UGARCHModel] | None, default=None
        Candidate univariate GARCH specifications to evaluate per asset.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `'univariate_model'`: list of optimal univariate model specifications per asset.

    Raises
    ------
    ValueError
        If `window_residuals_list` is empty or if no candidate univariate GARCH model
        converges for an asset series.
    """

    best_u_models = _tune_univariate_garch_per_asset(
        window_residuals_list, test_residuals_matrix, num_assets=num_assets, u_grid=u_grid
    )
    return {"univariate_model": best_u_models}


def _tune_dcc(
    window_residuals_list: list[NDArray[np.float64]],
    test_residuals_matrix: NDArray[np.float64],
    num_assets: int,
    u_grid: list[UGARCHModel] | None = None,
    asymmetric_grid: list[bool] | None = None,
) -> dict[str, Any]:
    """
    Tune Dynamic Conditional Correlation (DCC/aDCC) model via 2-stage optimization.

    Stage 1 tunes univariate GARCH specifications independently per asset equation.
    Stage 2 tunes multivariate correlation leverage dynamics (DCC vs aDCC) using rolling log-likelihood.
    If tuning cannot proceed or fails, defaults to the first value of candidate parameter lists.

    Parameters
    ----------
    window_residuals_list : list[NDArray[np.float64]]
        List of 2D in-sample mean residual matrices for each rolling window.
    test_residuals_matrix : NDArray[np.float64]
        2D matrix of shape `(K_valid, N)` containing out-of-sample mean test residuals.
    num_assets : int
        Total number of asset series.
    u_grid : list[UGARCHModel] | None, default=None
        Candidate univariate GARCH specifications to evaluate per asset in Stage 1.
    asymmetric_grid : list[bool] | None, default=None
        Candidate leverage options (`[False, True]`) to evaluate in Stage 2.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `'asymmetric'`, and `'univariate_model'`: list of univariate models.

    Raises
    ------
    ValueError
        If `window_residuals_list` is empty, if no candidate univariate GARCH model
        converges, or if no candidate DCC asymmetric specification converges.
    """

    num_windows = len(window_residuals_list)
    if num_windows == 0:
        raise ValueError("No rolling window residual data available for tuning DCC models")

    asym_options = list(asymmetric_grid) if asymmetric_grid is not None else _DEFAULT_ASYMMETRIC_GRID
    best_u_models = _tune_univariate_garch_per_asset(
        window_residuals_list, test_residuals_matrix, num_assets=num_assets, u_grid=u_grid
    )

    zero_mean_vec = np.zeros(num_assets, dtype=np.float64)
    best_asym = None
    best_total_ll = -np.inf

    for asym in asym_options:
        total_ll = 0.0
        valid_candidate = True

        for k in range(num_windows):
            window_res = window_residuals_list[k]
            test_res = test_residuals_matrix[k]

            try:
                cov_f, _ = predict_volatility(
                    window_res,
                    model="dcc",
                    asymmetric=asym,
                    univariate_model=best_u_models,
                )
                ll = multivariate_normal_log_likelihood(test_res, mean=zero_mean_vec, cov=cov_f)
                total_ll += float(ll)
            except Exception:
                valid_candidate = False
                break

        if valid_candidate and total_ll > best_total_ll:
            best_total_ll = total_ll
            best_asym = asym

    if best_asym is None or best_total_ll == -np.inf:
        raise ValueError("No valid DCC asymmetric specification converged across candidate grid")

    return {"asymmetric": best_asym, "univariate_model": best_u_models}


def _tune_dbekk(
    window_residuals_list: list[NDArray[np.float64]],
    test_residuals_matrix: NDArray[np.float64],
    num_assets: int,
    asymmetric_grid: list[bool] | None = None,
) -> dict[str, Any]:
    """
    Tune Diagonal BEKK (DBEKK) model leverage parameters via rolling multivariate log-likelihood.

    If tuning cannot proceed or fails, defaults to the first value of candidate parameter lists.

    Parameters
    ----------
    window_residuals_list : list[NDArray[np.float64]]
        List of 2D in-sample mean residual matrices for each rolling window.
    test_residuals_matrix : NDArray[np.float64]
        2D matrix of shape `(K_valid, N)` containing out-of-sample mean test residuals.
    num_assets : int
        Total number of asset series.
    asymmetric_grid : list[bool] | None, default=None
        Candidate leverage options (`[False, True]`) to evaluate.

    Returns
    -------
    dict[str, Any]
        Dictionary containing `'asymmetric'`: optimal leverage parameter choice.

    Raises
    ------
    ValueError
        If `window_residuals_list` is empty or if no candidate DBEKK asymmetric specification converges.
    """

    asym_options = list(asymmetric_grid) if asymmetric_grid is not None else _DEFAULT_ASYMMETRIC_GRID
    num_windows = len(window_residuals_list)

    if num_windows == 0:
        raise ValueError("No rolling window residual data available for tuning DBEKK models")

    zero_mean_vec = np.zeros(num_assets, dtype=np.float64)
    best_asym = None
    best_total_ll = -np.inf

    for asym in asym_options:
        total_ll = 0.0
        valid_candidate = True

        for k in range(num_windows):
            window_res = window_residuals_list[k]
            test_res = test_residuals_matrix[k]

            try:
                cov_f, _ = predict_volatility(window_res, model="dbekk", asymmetric=asym)
                ll = multivariate_normal_log_likelihood(test_res, mean=zero_mean_vec, cov=cov_f)
                total_ll += float(ll)
            except Exception:
                valid_candidate = False
                break

        if valid_candidate and total_ll > best_total_ll:
            best_total_ll = total_ll
            best_asym = asym

    if best_asym is None or best_total_ll == -np.inf:
        raise ValueError("No valid DBEKK asymmetric specification converged across candidate grid")

    return {"asymmetric": best_asym}


########################
### Tuning Functions ###
########################
def tune_mean_model(
    returns_matrix: NDArray[np.float64],
    model: Literal["naive", "ar", "var", "var_lasso"] = "naive",
    window_size: int = WINDOW_SIZE,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Tune optimal parameters for a mean forecasting model via rolling log-likelihood.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` where rows represent time periods and
        columns represent asset returns.
    model : {"naive", "ar", "var", "var_lasso"}, default="naive"
        The mean model to tune:

        * `'naive'`: Returns empty parameter dict (no tuning required).
        * `'ar'`: Tunes lag order `p` independently per asset equation.
        * `'var'`: Tunes joint lag order `p` (1..5) using OLS.
        * `'var_lasso'`: Tunes joint lag order `p` (1..20) and Lasso penalty `alpha`.
    window_size : int, default=WINDOW_SIZE
        Number of historical observations in each rolling evaluation window.
    **kwargs : Any
        Keyword arguments passed directly to the model tuner:

        * **p_grid** (*list[int]*, default=None): Custom candidate lag orders.
        * **alpha_grid** (*list[float]*, default=None): Custom Lasso penalty grid.

    Returns
    -------
    dict[str, Any]
        Dictionary of optimal hyperparameters for passing directly as `**kwargs` to `predict_mean`.

    Raises
    ------
    ValueError
        If `model` is unrecognized or arguments fail validation checks.
        Can also be raised if no candidate converges.
    """

    model_key = model.lower().replace("-", "_")

    if model_key == "naive":
        return {}
    if model_key == "ar":
        return _tune_ar_matrix(returns_matrix, window_size=window_size, **kwargs)
    if model_key == "var":
        return _tune_var(returns_matrix, window_size=window_size, **kwargs)
    if model_key == "var_lasso":
        return _tune_var_lasso(returns_matrix, window_size=window_size, **kwargs)

    raise ValueError(f"unknown model '{model}', supported models are 'naive', 'ar', 'var', 'var_lasso'")


def tune_volatility_model(
    returns_matrix: NDArray[np.float64],
    model: VolatilityModel = "dcc",
    mean_model: MeanModel = "naive",
    window_size: int = WINDOW_SIZE,
    mean_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Tune optimal hyperparameters for a multivariate volatility model via rolling log-likelihood.

    Demeans raw return series sequentially per rolling window using the mean model to construct aligned
    in-sample residual matrices and out-of-sample test residuals. If no rolling windows succeed or if
    tuning fails for all candidates, defaults to the first value from the candidate parameter lists.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        2D array of shape `(T, N)` where rows represent time periods and
        columns represent asset returns.
    model : VolatilityModel, default="dcc"
        The multivariate GARCH volatility model to tune:

        * `'naive'`: Sample covariance matrix (no parameter tuning required).
        * `'ccc'`: Tunes univariate GARCH per equation (Stage 1).
        * `'dcc'`: Tunes univariate GARCH per equation (Stage 1) and asymmetric leverage parameter (Stage 2).
        * `'dbekk'`: Tunes asymmetric leverage parameter matrix via rolling multivariate log-likelihood.
        * `'go_garch'`: No parameter tuning.
    mean_model : MeanModel, default="naive"
        The mean model used to compute rolling mean forecasts and residuals.
    window_size : int, default=WINDOW_SIZE
        Number of historical observations in each rolling evaluation window.
    mean_kwargs : dict[str, Any] | None, default=None
        Hyperparameters passed directly to `predict_mean` (e.g., `p`, `p_list`, `alpha`).
    **kwargs : Any
        Keyword arguments passed directly to the volatility tuner:

        * **u_grid** (*list[UGARCHModel]*, default=None): Custom candidate univariate GARCH specifications.
        * **asymmetric_grid** (*list[bool]*, default=None): Custom candidate leverage choices (`[False, True]`).

    Returns
    -------
    dict[str, Any]
        Dictionary of optimal hyperparameters for passing directly as `**kwargs` to `predict_volatility`.

    Raises
    ------
    ValueError
        If `model` is unrecognized or if inputs fail initial shape/window validation checks.
        Can also be raised if no candidate converges.
    """

    _, num_assets = _validate_returns_matrix(returns_matrix, window_size)
    model_key = model.lower().replace("-", "_")

    if model_key in ("naive", "go_garch"):
        return {}

    # Sequentially fit mean model on rolling windows to generate aligned residual arrays
    window_residuals_list, test_residuals_matrix = _generate_rolling_residuals(
        returns_matrix,
        mean_model=mean_model,
        window_size=window_size,
        mean_kwargs=mean_kwargs,
    )

    if model_key == "ccc":
        return _tune_ccc(window_residuals_list, test_residuals_matrix, num_assets=num_assets, **kwargs)
    if model_key == "dcc":
        return _tune_dcc(window_residuals_list, test_residuals_matrix, num_assets=num_assets, **kwargs)
    if model_key == "dbekk":
        return _tune_dbekk(window_residuals_list, test_residuals_matrix, num_assets=num_assets, **kwargs)

    raise ValueError(
        f"unknown volatility model '{model}', supported models are 'naive', 'ccc', 'dcc', 'dbekk', 'go_garch'"
    )
