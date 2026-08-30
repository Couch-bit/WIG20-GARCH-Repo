import warnings
from typing import Any, Literal, cast, get_args

import numpy as np
import rpy2.rinterface_lib.callbacks as rpy2_callbacks
import rpy2.robjects as robjects
from numpy.typing import NDArray
from rpy2.robjects import numpy2ri
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression

#############
### Setup ###
#############
# Activate automatic NumPy <-> R array conversion
numpy2ri.activate()

# Suppress console output from R
rpy2_callbacks.consolewrite_warnerror = lambda x: None


# Define helper functions in R environment
robjects.r("""
library(rugarch)
library(rmgarch)
library(BEKKs)

extract_H_t <- function(H_obj, t_idx, T_obs, N_val) {
  if (is.list(H_obj)) return(H_obj[[t_idx]])
  d <- dim(H_obj)
  if (length(d) == 3) {
    return(if (d[3] == T_obs) H_obj[,, t_idx] else H_obj[t_idx,, ])
  } else if (length(d) == 2) {
    return(matrix(if (d[1] == T_obs) H_obj[t_idx, ] else H_obj[, t_idx], nrow = N_val, ncol = N_val))
  }
  stop("Unknown covariance matrix structure in fit$H")
}

r_predict_ugarch <- function(series, model_type) {
  spec <- ugarchspec(
    variance.model = list(model = model_type, garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
  )
  fit <- ugarchfit(spec = spec, data = series, solver = "hybrid")
  
  if (fit@fit$convergence != 0) {
    conv_code <- fit@fit$convergence
    stop(paste("Univariate GARCH model (", model_type, ") failed to converge with code ", conv_code, sep = ""))
  }
  
  fcst <- ugarchforecast(fit, n.ahead = 1)
  var_f <- as.numeric(sigma(fcst)^2)
  std_res <- as.numeric(residuals(fit, standardize = TRUE))
  
  list(var_forecast = var_f, std_residuals = std_res)
}

r_predict_ccc <- function(data_mat, u_models) {
  N <- ncol(data_mat)
  if (length(u_models) == 1) {
    u_models <- rep(u_models, N)
  }
  
  uspec_list <- lapply(u_models, function(m) {
    ugarchspec(
      variance.model = list(model = m, garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
  })
  
  mspec <- multispec(uspec_list)
  u_fit <- multifit(mspec, data = data_mat)
  
  u_conv <- sapply(u_fit@fit, function(x) x@fit$convergence)
  if (any(u_conv != 0)) {
    failures <- which(u_conv != 0)
    stop(paste("CCC Stage 1 (Univariate GARCH) failed to converge for asset(s):", paste(failures, collapse = ", ")))
  }
  
  Z_mat <- do.call(cbind, lapply(u_fit@fit, function(x) as.numeric(residuals(x, standardize = TRUE))))
  colnames(Z_mat) <- colnames(data_mat)
  
  R_mat <- cor(Z_mat)
  
  eig <- eigen(R_mat)
  R_inv_sqrt <- eig$vectors %*% diag(1 / sqrt(pmax(eig$values, 1e-8))) %*% t(eig$vectors)
  std_res <- Z_mat %*% R_inv_sqrt
  colnames(std_res) <- colnames(data_mat)
  
  u_fcst <- multiforecast(u_fit, n.ahead = 1)
  sigmas <- as.numeric(sigma(u_fcst))
  cov_f <- diag(sigmas) %*% R_mat %*% diag(sigmas)
  colnames(cov_f) <- colnames(data_mat)
  rownames(cov_f) <- colnames(data_mat)
  
  list(cov_forecast = cov_f, std_residuals = std_res)
}

r_predict_dcc <- function(data_mat, asymmetric, u_models) {
  N <- ncol(data_mat)
  T_obs <- nrow(data_mat)
  if (length(u_models) == 1) {
    u_models <- rep(u_models, N)
  }
  
  uspec_list <- lapply(u_models, function(m) {
    ugarchspec(
      variance.model = list(model = m, garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
  })
  
  mspec <- multispec(uspec_list)
  u_fit <- multifit(mspec, data = data_mat)
  
  u_conv <- sapply(u_fit@fit, function(x) x@fit$convergence)
  if (any(u_conv != 0)) {
    failures <- which(u_conv != 0)
    stop(paste("DCC Stage 1 (Univariate GARCH) failed to converge for asset(s):", paste(failures, collapse = ", ")))
  }
  
  Z_mat <- do.call(cbind, lapply(u_fit@fit, function(x) as.numeric(residuals(x, standardize = TRUE))))
  
  model_type <- if (asymmetric) "aDCC" else "DCC"
  dspec <- dccspec(mspec, dccOrder = c(1, 1), model = model_type)
  fit <- dccfit(dspec, data = data_mat, fit = u_fit)
  
  m_conv <- fit@mfit$convergence
  if (!is.null(m_conv) && m_conv != 0) {
    stop(paste("DCC Stage 2 (Multivariate Correlation) failed to converge with code", m_conv))
  }
  
  R_array <- rcor(fit)
  std_res <- matrix(0, nrow = T_obs, ncol = N)
  
  for (t in 1:T_obs) {
    R_t <- R_array[,, t]
    eig <- eigen(R_t)
    R_inv_sqrt <- eig$vectors %*% diag(1 / sqrt(pmax(eig$values, 1e-8))) %*% t(eig$vectors)
    std_res[t, ] <- as.numeric(R_inv_sqrt %*% Z_mat[t, ])
  }
  colnames(std_res) <- colnames(data_mat)
  
  fcst <- dccforecast(fit, n.ahead = 1)
  cov_f <- rcov(fcst)[[1]][,,1]
  
  list(cov_forecast = cov_f, std_residuals = std_res)
}

r_predict_dbekk <- function(data_mat, asymmetric) {
  fit <- tryCatch({
    spec <- bekk_spec(model = list(type = "dbekk", asymmetric = asymmetric))
    bekk_fit(spec, data = data_mat)
  }, error = function(e) {
    stop(paste("DBEKK model failed to converge or fit:", e$message))
  })

  if (!is.null(fit$BEKK_valid) && !fit$BEKK_valid) {
    stop("DBEKK model parameters are not valid")
  }
  
  T_obs <- nrow(data_mat)
  N <- ncol(data_mat)

  C_mat <- fit$C0 %*% t(fit$C0)
  A_mat <- if (is.list(fit$A)) fit$A[[1]] else fit$A
  G_mat <- if (is.list(fit$G)) fit$G[[1]] else fit$G
  
  # Use raw input matrix directly under zero-mean assumption
  e_T <- matrix(data_mat[T_obs, ], ncol = 1)
  H_T <- extract_H_t(fit$H, T_obs, T_obs, N)
  
  # Base 1-step ahead forecast: H_{T+1} = C + A' (e_T e_T') A + G' H_T G
  cov_f <- C_mat + t(A_mat) %*% (e_T %*% t(e_T)) %*% A_mat + t(G_mat) %*% H_T %*% G_mat
  
  # Add leverage term using B matrix if asymmetric: + B' (eta_T eta_T') B
  if (asymmetric && !is.null(fit$B)) {
    B_mat <- if (is.list(fit$B)) fit$B[[1]] else fit$B
    eta_T <- pmin(e_T, 0)
    cov_f <- cov_f + t(B_mat) %*% (eta_T %*% t(eta_T)) %*% B_mat
  }
  
  # Compute standardized residuals manually: z_t = H_t^{-1/2} * e_t
  std_res <- matrix(0, nrow = T_obs, ncol = N)
  for (t in 1:T_obs) {
    H_t <- extract_H_t(fit$H, t, T_obs, N)
    e_t <- matrix(data_mat[t, ], ncol = 1)
    
    eig <- eigen(H_t)
    inv_sqrt_H <- eig$vectors %*% diag(1 / sqrt(pmax(eig$values, 1e-8))) %*% t(eig$vectors)
    std_res[t, ] <- as.numeric(inv_sqrt_H %*% e_t)
  }
  colnames(std_res) <- colnames(data_mat)
  
  list(cov_forecast = cov_f, std_residuals = std_res)
}

r_predict_gogarch <- function(data_mat, u_models) {
  N <- ncol(data_mat)
  T_obs <- nrow(data_mat)
  if (length(u_models) == 1) {
    u_models <- rep(u_models, N)
  }
  
  uspec_list <- lapply(u_models, function(m) {
    ugarchspec(
      variance.model = list(model = m, garchOrder = c(1, 1)),
      mean.model = list(armaOrder = c(0, 0), include.mean = FALSE)
    )
  })
  
  mspec <- multispec(uspec_list)
  gspec <- gogarchspec(mspec, mean.model = list(model = "constant"))

  # This is helpful for removing messages from fitting
  temp_file <- tempfile()
  sink(temp_file)
  fit <- tryCatch({
    gogarchfit(gspec, data = data_mat)
  }, finally = {
    sink()
    unlink(temp_file)
  })

  # Univariate convergence check
  u_conv <- sapply(fit@mfit$ufit@fit, function(x) x@fit$convergence)
  if (any(u_conv != 0)) {
    failures <- which(u_conv != 0)
    stop(paste("Univariate GARCH failed to converge for components:", paste(failures, collapse = ", ")))
  }
  
  # Subtract estimated constant mean vector (r_t - hat{mu}) as required by GO-GARCH
  raw_asset_res <- data_mat - fit@mfit$mu
  
  # Full covariance whitening using asset covariance matrices H_t
  H_array <- rcov(fit)
  std_res <- matrix(0, nrow = T_obs, ncol = N)
  
  for (t in 1:T_obs) {
    H_t <- H_array[,, t]
    e_t <- matrix(raw_asset_res[t, ], ncol = 1)
    
    eig <- eigen(H_t)
    inv_sqrt_H <- eig$vectors %*% diag(1 / sqrt(pmax(eig$values, 1e-8))) %*% t(eig$vectors)
    std_res[t, ] <- as.numeric(inv_sqrt_H %*% e_t)
  }
  colnames(std_res) <- colnames(data_mat)
  
  fcst <- gogarchforecast(fit, n.ahead = 1)
  cov_f <- rcov(fcst)[[1]][,,1]
  
  list(cov_forecast = cov_f, std_residuals = std_res)
}
""")

MeanModel = Literal["naive", "ar", "var", "var_lasso"]
UGARCHModel = Literal["sGARCH", "eGARCH", "gjrGARCH"]
VolatilityModel = Literal["naive", "ccc", "dcc", "dbekk", "go_garch"]

_ALLOWED_MODELS = set(get_args(UGARCHModel))


#########################
### Univariate Models ###
#########################
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


def predict_univariate_garch(
    series: NDArray[np.float64],
    model: UGARCHModel | str = "sGARCH",
) -> tuple[float, NDArray[np.float64]]:
    """
    Fit a 1-lag univariate GARCH model to zero-mean residuals using R's ``rugarch``.

    Parameters
    ----------
    series : NDArray[np.float64]
        A 1D array of shape ``(T,)`` containing zero-mean residuals from a mean model.
    model : UGARCHModel | str, default="sGARCH"
        R ``rugarch`` model specification. Must be one of ``'sGARCH'`` (Standard GARCH),
        ``'eGARCH'`` (Exponential GARCH), or ``'gjrGARCH'`` (GJR-GARCH).

    Returns
    -------
    var_forecast : float
        The 1-step ahead conditional variance forecast.
    std_residuals : NDArray[np.float64]
        A 1D array of shape ``(T,)`` containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If ``series`` is invalid, if ``model`` is unsupported, or if model fitting
        fails to converge.
    """

    if series.ndim != 1 or len(series) < 10:
        raise ValueError("series must be a 1D array with at least 10 observations")
    if model not in _ALLOWED_MODELS:
        raise ValueError(f"invalid model '{model}', supported models are 'sGARCH', 'eGARCH', 'gjrGARCH'")

    try:
        res = robjects.r["r_predict_ugarch"](series, model)
        var_f = float(np.asarray(res.rx2("var_forecast"))[0])
        std_res = np.asarray(res.rx2("std_residuals"), dtype=np.float64)
        return var_f, std_res
    except Exception as e:
        raise ValueError(f"Univariate GARCH estimation failed: {e}") from e


############################
### Multivariate Helpers ###
############################
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


def _predict_naive_cov(
    returns_matrix: NDArray[np.float64],
    cov_eps: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Calculate 1-step ahead naive sample covariance forecast and standardized residuals.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape (T, N) containing zero-mean residuals from a mean model.
    cov_eps : float, default=1e-8
        Minimum eigenvalue floor to prevent numerical instability during matrix square root inverse.

    Returns
    -------
    cov_forecast : NDArray[np.float64]
        A 2D array of shape (N, N) containing the sample covariance matrix forecast.
    std_residuals : NDArray[np.float64]
        A 2D array of shape (T, N) containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If returns_matrix is not 2D or has less than 10 observations.
    """

    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 10:
        raise ValueError("returns_matrix must be a 2D array with at least 10 rows (observations)")

    t_obs, _ = returns_matrix.shape
    cov_f = (returns_matrix.T @ returns_matrix) / t_obs

    # Spectral decomposition for matrix square root inverse H^{-1/2}
    eig_val, eig_vec = np.linalg.eigh(cov_f)
    inv_sqrt_cov = eig_vec @ np.diag(1.0 / np.sqrt(np.maximum(eig_val, cov_eps))) @ eig_vec.T
    std_res = returns_matrix @ inv_sqrt_cov

    return cov_f, std_res


def _validate_u_models(
    u_model: UGARCHModel | list[UGARCHModel],
    num_assets: int,
) -> list[UGARCHModel]:
    """
    Validate and format univariate GARCH model specification choices.

    Parameters
    ----------
    u_model : UGARCHModel | list[UGARCHModel]
        A single model string or a list of model strings for each asset.
    num_assets : int
        The number of assets (columns) in the returns matrix.

    Returns
    -------
    list[str]
        A list of validated model string identifiers matching R ``rugarch`` syntax.

    Raises
    ------
    ValueError
        If any model string is invalid or if sequence length does not equal 1 or ``num_assets``.
    """

    if isinstance(u_model, str):
        models = [u_model]
    else:
        models = list(u_model)

    for m in models:
        if m not in _ALLOWED_MODELS:
            raise ValueError(f"invalid model '{m}'")

    if len(models) != 1 and len(models) != num_assets:
        raise ValueError(
            f"length of univariate_model list ({len(models)}) must be 1 or match number of assets ({num_assets})"
        )

    return models


def _predict_ccc(
    returns_matrix: NDArray[np.float64],
    univariate_model: UGARCHModel | list[UGARCHModel] = "sGARCH",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead Constant Conditional Correlation (CCC-GARCH) covariance forecast.

    Fits univariate GARCH(1,1) models to each asset series and combines them using a constant
    conditional correlation matrix via R's ``rmgarch`` package.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` where rows represent time periods and
        columns represent asset residuals from a mean model.
    univariate_model : UGARCHModel | list[UGARCHModel], default="sGARCH"
        Univariate GARCH model specification(s). Can be a single model string applied
        to all assets or a list of model strings matching the number of assets ``N``.
        Supported options are ``'sGARCH'``, ``'eGARCH'``, and ``'gjrGARCH'``.

    Returns
    -------
    cov_forecast : NDArray[np.float64]
        A 2D array of shape ``(N, N)`` containing the 1-step ahead conditional covariance
        matrix forecast.
    std_residuals : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If ``returns_matrix`` is invalid, if ``univariate_model`` fails validation, or if
        Stage 1 (univariate) or Stage 2 (multivariate) optimization fails to converge.
    """

    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 10:
        raise ValueError("returns_matrix must be a 2D array with at least 10 rows (observations)")

    num_assets = returns_matrix.shape[1]
    u_models = _validate_u_models(univariate_model, num_assets)

    try:
        res = robjects.r["r_predict_ccc"](returns_matrix, u_models)
        cov_f = np.asarray(res.rx2("cov_forecast"), dtype=np.float64)
        std_res = np.asarray(res.rx2("std_residuals"), dtype=np.float64)
        return cov_f, std_res
    except Exception as e:
        raise ValueError(f"CCC estimation failed: {e}") from e


def _predict_dcc(
    returns_matrix: NDArray[np.float64],
    asymmetric: bool = False,
    univariate_model: UGARCHModel | list[UGARCHModel] = "sGARCH",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead Dynamic Conditional Correlation (DCC or ADCC) covariance forecast.

    Fits univariate GARCH(1,1) models to each asset series and estimates time-varying
    conditional correlations via R's ``rmgarch`` package.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` where rows represent time periods and
        columns represent asset residuals from a mean model.
    asymmetric : bool, default=False
        If ``True``, estimates an Asymmetric DCC (aDCC) model incorporating leverage
        effects in conditional correlation dynamics.
    univariate_model : UGARCHModel | list[UGARCHModel], default="sGARCH"
        Univariate GARCH model specification(s). Can be a single model string applied
        to all assets or a sequence of model strings matching the number of assets ``N``.
        Supported options are ``'sGARCH'``, ``'eGARCH'``, and ``'gjrGARCH'``.

    Returns
    -------
    cov_forecast : NDArray[np.float64]
        A 2D array of shape ``(N, N)`` containing the 1-step ahead conditional covariance
        matrix forecast.
    std_residuals : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If ``returns_matrix`` is invalid, if ``univariate_model`` fails validation, or if
        Stage 1 (univariate) or Stage 2 (multivariate) optimization fails to converge.
    """

    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 10:
        raise ValueError("returns_matrix must be a 2D array with at least 10 rows (observations)")

    num_assets = returns_matrix.shape[1]
    u_models = _validate_u_models(univariate_model, num_assets)

    try:
        res = robjects.r["r_predict_dcc"](returns_matrix, asymmetric, u_models)
        cov_f = np.asarray(res.rx2("cov_forecast"), dtype=np.float64)
        std_res = np.asarray(res.rx2("std_residuals"), dtype=np.float64)
        return cov_f, std_res
    except Exception as e:
        raise ValueError(f"DCC estimation failed: {e}") from e


def _predict_dbekk(
    returns_matrix: NDArray[np.float64],
    asymmetric: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead Diagonal BEKK (DBEKK) covariance forecast.

    Fits a Diagonal BEKK multivariate GARCH model via R's ``BEKKs`` package. Standardized
    residuals are computed manually using spectral matrix
    decomposition of fitted covariance matrices.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` where rows represent time periods and
        columns represent asset residuals from a mean model.
    asymmetric : bool, default=False
        If ``True``, incorporates asymmetric leverage term matrix into the
        BEKK variance recursion equation.

    Returns
    -------
    cov_forecast : NDArray[np.float64]
        A 2D array of shape ``(N, N)`` containing the 1-step ahead conditional covariance
        matrix forecast.
    std_residuals : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If ``returns_matrix`` is invalid or if solver optimization status indicates non-convergence.
    """

    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 10:
        raise ValueError("returns_matrix must be a 2D array with at least 10 rows (observations)")

    try:
        res = robjects.r["r_predict_dbekk"](returns_matrix, asymmetric)
        cov_f = np.asarray(res.rx2("cov_forecast"), dtype=np.float64)
        std_res = np.asarray(res.rx2("std_residuals"), dtype=np.float64)
        return cov_f, std_res
    except Exception as e:
        raise ValueError(f"DBEKK estimation failed: {e}") from e


def _predict_go_garch(
    returns_matrix: NDArray[np.float64],
    univariate_model: UGARCHModel | list[UGARCHModel] = "sGARCH",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead Generalized Orthogonal GARCH (GO-GARCH) covariance forecast.

    Decomposes the multivariate return series into independent components using Independent
    Component Analysis (ICA) and models component dynamics via R's ``rmgarch`` package.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` where rows represent time periods and
        columns represent asset residuals from a mean model.
    univariate_model : UGARCHModel | list[UGARCHModel], default="sGARCH"
        Univariate GARCH model specification(s) for ICA components. Can be a single model
        string applied to all components or a sequence matching the number of components ``N``.
        Supported options are ``'sGARCH'``, ``'eGARCH'``, and ``'gjrGARCH'``.

    Returns
    -------
    cov_forecast : NDArray[np.float64]
        A 2D array of shape ``(N, N)`` containing the 1-step ahead conditional covariance
        matrix forecast.
    std_residuals : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If ``returns_matrix`` is invalid, if ``univariate_model`` fails validation, or if
        Stage 1 (ICA component GARCH) or Stage 2 (mixing matrix) optimization fails to converge.
    """

    if returns_matrix.ndim != 2 or returns_matrix.shape[0] < 10:
        raise ValueError("returns_matrix must be a 2D array with at least 10 rows (observations)")

    num_assets = returns_matrix.shape[1]
    u_models = _validate_u_models(univariate_model, num_assets)

    try:
        res = robjects.r["r_predict_gogarch"](returns_matrix, u_models)
        cov_f = np.asarray(res.rx2("cov_forecast"), dtype=np.float64)
        std_res = np.asarray(res.rx2("std_residuals"), dtype=np.float64)
        return cov_f, std_res
    except Exception as e:
        raise ValueError(f"GO-GARCH estimation failed: {e}") from e


###########################
### Multivariate Models ###
###########################
def predict_mean(
    returns_matrix: NDArray[np.float64],
    model: MeanModel,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead return forecast using the specified mean model.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape (T, N) where rows represent time periods and
        columns represent asset returns.
    model : MeanModel
        The forecasting model to use:

        * `'naive'`: Historical column means.
        * `'ar'`: Univariate AR(p) models fitted independently per asset.
        * `'var'`: Vector Autoregressive VAR(p) model.
        * `'var_lasso'`: Vector Autoregressive VAR(p) model with Lasso regularization.
    **kwargs : Any
        Keyword arguments passed directly to the underlying model implementation:

        * **p** (*int*, default=1): Lag order for `'var'` or single lag order for `'ar'`.
        * **p_list** (*list[int] | int*, default=1): Lag order per asset or single integer for `'ar'`.
        * **alpha** (*float*, default=1.0): Lasso penalty parameter for `'var'`.
        * **tol** (*float*, default=1e-4): Convergence tolerance for `'var'` Lasso solver.
        * **max_iter** (*int*, default=10000): Maximum iterations for `'var'` Lasso solver.

    Returns
    -------
    forecasts : NDArray[np.float64]
        A 1D array of shape (N,) containing 1-step ahead return forecasts.
    residuals : NDArray[np.float64]
        A 2D array of shape (T_eff, N) containing in-sample residuals.

    Raises
    ------
    ValueError
        If model is unrecognized, arguments are invalid, or optimization fails.
    """

    if model == "naive":
        return _predict_naive_mean(returns_matrix)
    if model == "ar":
        return _predict_ar_matrix(returns_matrix, **kwargs)
    if model in ("var", "var_lasso"):
        return _predict_var_lasso(returns_matrix, **kwargs)

    raise ValueError(f"unknown model '{model}'")


def predict_volatility(
    returns_matrix: NDArray[np.float64],
    model: VolatilityModel,
    **kwargs: Any,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Calculate 1-step ahead covariance forecast and standardized residuals using a volatility model.

    Parameters
    ----------
    returns_matrix : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` where rows represent time periods and
        columns represent asset residuals from a mean model.
    model : VolatilityModel
        The multivariate GARCH forecasting model to use:

        * `'naive'`: Sample covariance matrix assuming zero-mean residuals.
        * `'ccc'`: Constant Conditional Correlation GARCH.
        * `'dcc'`: Dynamic Conditional Correlation GARCH (or aDCC if asymmetric).
        * `'dbekk'`: Diagonal BEKK GARCH (can be asymmetric).
        * `'go_garch'`: Generalized Orthogonal GARCH via ICA decomposition.
    **kwargs : Any
        Keyword arguments passed directly to the underlying model implementation:

        * **asymmetric** (*bool*, default=False): Incorporate leverage effects (for `'dcc'` and `'dbekk'`).
        * **univariate_model** (*UGARCHModel | list[UGARCHModel]*, default="sGARCH"): Univariate GARCH name in rugarch.

    Returns
    -------
    cov_forecast : NDArray[np.float64]
        A 2D array of shape ``(N, N)`` containing the 1-step ahead conditional covariance forecast.
    std_residuals : NDArray[np.float64]
        A 2D array of shape ``(T, N)`` containing in-sample standardized residuals.

    Raises
    ------
    ValueError
        If model is unrecognized, parameters fall outside valid ranges, or optimization fails.
    """

    model_key = model.lower().replace("-", "_")

    if model_key == "naive":
        return _predict_naive_cov(returns_matrix, **kwargs)
    if model_key == "ccc":
        return _predict_ccc(returns_matrix, **kwargs)
    if model_key == "dcc":
        return _predict_dcc(returns_matrix, **kwargs)
    if model_key == "dbekk":
        return _predict_dbekk(returns_matrix, **kwargs)
    if model_key == "go_garch":
        return _predict_go_garch(returns_matrix, **kwargs)

    raise ValueError(f"unknown volatility model '{model}'")
