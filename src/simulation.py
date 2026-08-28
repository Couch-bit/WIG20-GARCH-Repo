import json
import logging
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from metrics import MetricName, compute_metric, multivariate_normal_log_likelihood
from model_optimization import tune_mean_model, tune_volatility_model
from models import MeanModel, VolatilityModel, predict_mean, predict_volatility
from portfolio_optimization import optimize_portfolio


##########################
### Simulation Helpers ###
##########################
def _generate_fhs_returns(
    forecast_mean: NDArray[np.float64],
    forecast_cov: NDArray[np.float64],
    std_residuals: NDArray[np.float64],
    eig_eps: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Generate simulated returns using Filtered Historical Simulation (FHS).

    Parameters
    ----------
    forecast_mean : NDArray[np.float64]
        1D array of shape (N,) containing the 1-step ahead mean forecast.
    forecast_cov : NDArray[np.float64]
        2D array of shape (N, N) containing the 1-step ahead covariance forecast.
    std_residuals : NDArray[np.float64]
        2D array of shape (T, N) containing the historical standardized residuals.
    eig_eps : float, default=1e-8
        Minimal value for eigenvalues used for spectral decomposition to be enforced for numerical stability.

    Returns
    -------
    NDArray[np.float64]
        2D array of shape (T, N) containing simulated asset log-returns.
    """

    # Calculate symmetric matrix square root of forecasted covariance
    eig_val, eig_vec = np.linalg.eigh(forecast_cov)
    eig_val = np.maximum(eig_val, eig_eps)  # Prevent numerical instability
    sqrt_cov = eig_vec @ np.diag(np.sqrt(eig_val)) @ eig_vec.T

    # Simulated log-returns: mu + z * sqrt(H)
    simulated_returns = cast(NDArray[np.float64], forecast_mean + std_residuals @ sqrt_cov)

    return simulated_returns


###########################
### Simulation Function ###
###########################
def run_backtest(
    returns_df: pd.DataFrame,
    window_size: int,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    mean_model: MeanModel,
    volatility_model: VolatilityModel,
    optimize_portfolio_flag: bool,
    ga_metric: MetricName,
    save_path: str | Path,
    rf_rates_path: str | Path | None = None,
    log_path: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Run rolling-window simulation, fit models, optionally optimize a portfolio, and compute metrics.

    Parameters
    ----------
    returns_df : pd.DataFrame
        DataFrame of daily log-returns with a datetime index and assets as columns.
    window_size : int
        Number of historical observations in the rolling estimation window.
    start_date : str | pd.Timestamp
        The start date for the simulation period.
    end_date : str | pd.Timestamp
        The end date for the simulation period.
    mean_model : MeanModel
        String identifier for the mean forecasting model.
    volatility_model : VolatilityModel
        String identifier for the multivariate volatility model.
    optimize_portfolio_flag : bool
        Whether to calculate optimal portfolio weights using FHS and the genetic algorithm.
    ga_metric : MetricName
        String identifier for the optimization metric target used in the genetic algorithm.
    save_path : str | Path
        Path where the resulting DataFrame should be saved as a Parquet file.
    rf_rates_path : str | Path | None, default=None
        Path to a JSON file containing monthly risk-free rates indexed by 'YYYY-MM'.
    log_path : str | Path | None, default=None
        Path to a log file where execution warnings and model failures are recorded.
    **kwargs : Any
        Additional keyword arguments passed directly to the genetic algorithm.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing daily log-likelihoods, optimal weights, simulation metrics,
        portfolio returns, and portfolio turnover.

    Raises
    ------
    ValueError
        If inputs fail validation or if required historical/future periods are out of bounds.
    """

    # Validation checks
    if window_size <= 0:
        raise ValueError(f"window_size must be strictly positive, got {window_size}")

    # Configure logger
    logger = logging.getLogger("backtest_logger")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()

    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    # Load risk-free rates dictionary from JSON if path is provided
    rf_rates_dict: dict[str, float] = {}
    if rf_rates_path is not None:
        rates_file = Path(rf_rates_path)
        if not rates_file.exists():
            raise ValueError(f"Risk-free rates JSON file not found at path: {rf_rates_path}")
        with rates_file.open("r", encoding="utf-8") as f:
            rf_rates_dict = json.load(f)

    df = returns_df.copy()
    df.index = pd.to_datetime(df.index)

    start_idx = df.index.get_indexer(pd.DatetimeIndex([pd.to_datetime(start_date)]), method="bfill")[0]
    end_idx = df.index.get_indexer(pd.DatetimeIndex([pd.to_datetime(end_date)]), method="ffill")[0]

    tuning_window = int(np.floor(window_size * 1.5))

    # Validate index boundaries
    if start_idx - tuning_window < 0:
        raise ValueError(
            f"Not enough historical data before start_date '{start_date}'. Need {tuning_window} days for tuning window"
        )

    results_list = []
    mean_kwargs: dict[str, Any] = {}
    vol_kwargs: dict[str, Any] = {}
    prev_valid_assets: list[str] = []
    prev_full_weights = pd.Series(0.0, index=df.columns)
    prev_test_date = None
    prev_population: list[list[int]] | None = None

    for test_idx in tqdm(range(start_idx, end_idx + 1), desc="Running backtest"):
        current_t = test_idx - 1  # Last day of history
        test_date = df.index[test_idx]
        date_str = test_date.strftime("%Y-%m-%d")

        is_first_day = test_idx == start_idx
        is_new_year = (test_date.year != prev_test_date.year) if prev_test_date is not None else False
        prev_test_date = test_date

        # Extract estimation window, removing columns with NaNs in the period (including test day)
        window_slice = df.iloc[current_t - window_size + 1 : test_idx + 1]
        valid_cols = window_slice.columns[window_slice.notna().all(axis=0)]
        valid_assets_list = list(valid_cols)

        if len(valid_cols) < 2:
            raise ValueError(f"[{date_str}] Less than 2 valid assets available without NaNs in the estimation window")

        # TUNING PHASE
        if is_first_day or is_new_year:
            tuning_slice = df.iloc[current_t - tuning_window + 1 : current_t + 1]
            valid_tune_cols = tuning_slice.columns[tuning_slice.notna().all(axis=0)]

            if len(valid_tune_cols) >= 2:
                tuning_ret_mat = tuning_slice[valid_tune_cols].values

                # Tune mean model and capture emitted warnings
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    try:
                        mean_kwargs = tune_mean_model(tuning_ret_mat, model=mean_model, window_size=window_size)
                    except Exception as e:
                        logger.warning(f"[{date_str}] [Mean Model Tuning] Failed for '{mean_model}': {e}")
                        mean_kwargs = {}

                    for w in caught_warnings:
                        logger.warning(f"[{date_str}] [Mean Model Tuning] {w.message}")

                # Tune volatility model and capture emitted warnings
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    try:
                        vol_kwargs = tune_volatility_model(
                            tuning_ret_mat,
                            model=volatility_model,
                            mean_model=mean_model,
                            window_size=window_size,
                            mean_kwargs=mean_kwargs,
                        )
                    except Exception as e:
                        logger.warning(f"[{date_str}] [Volatility Model Tuning] Failed for '{volatility_model}': {e}")
                        vol_kwargs = {}

                    for w in caught_warnings:
                        logger.warning(f"[{date_str}] [Volatility Model Tuning] {w.message}")
            else:
                logger.warning(f"[{date_str}] [Tuning] Tuning couldn't run due to less than 2 valid assets")

        # ESTIMATION PHASE
        window_data = window_slice[valid_cols].iloc[:-1].values
        test_obs = cast(NDArray[np.float64], window_slice[valid_cols].iloc[-1].values)

        failed_fit = False
        try:
            fcst_mean, in_res = predict_mean(window_data, model=mean_model, **mean_kwargs)
        except Exception as e:
            logger.warning(f"[{date_str}] Mean model execution failed for '{mean_model}': {e}")
            failed_fit = True

        if not failed_fit:
            try:
                fcst_cov, std_res = predict_volatility(in_res, model=volatility_model, **vol_kwargs)
            except Exception as e:
                logger.warning(f"[{date_str}] Volatility model execution failed for '{volatility_model}': {e}")
                failed_fit = True

        if failed_fit:
            logger.warning(f"[{date_str}] Falling back to naive model optimization")
            fcst_mean, in_res = predict_mean(window_data, model="naive")
            fcst_cov, std_res = predict_volatility(in_res, model="naive")

        ll = float(multivariate_normal_log_likelihood(test_obs, mean=fcst_mean, cov=fcst_cov))

        # Setup Default Placeholder Metrics
        current_weights = np.zeros(len(valid_cols))
        best_val = np.nan
        is_finite = False
        port_simple_ret = np.nan
        turnover = np.nan
        full_weights = pd.Series(0.0, index=df.columns)

        # PORTFOLIO OPTIMIZATION PHASE
        if optimize_portfolio_flag:
            valid_changed = prev_valid_assets != valid_assets_list

            # If the set of active assets changed, invalidate warm start population
            if valid_changed:
                prev_population = None

            # Extract monthly risk-free rate for current evaluation date
            year_month_key = test_date.strftime("%Y-%m")
            rf_rate = float(rf_rates_dict.get(year_month_key, 0.0))

            # Generate simulated returns and convert them into simple returns upfront for GA efficiency
            fhs_returns = _generate_fhs_returns(fcst_mean, fcst_cov, std_res)
            fhs_simple_returns = np.exp(fhs_returns) - 1.0
            fhs_excess_returns = fhs_simple_returns - rf_rate

            def ga_eval_func(ret_mat: NDArray[np.float64], w: NDArray[np.float64]) -> float:
                port_excess = np.sum(ret_mat * w, axis=1)
                return compute_metric(port_excess, metric=ga_metric)

            # Run GA if it's the first day, investments changed, the fit succeeded, or there are no fallback weights
            if is_first_day or valid_changed or not failed_fit or prev_full_weights.sum() == 0:
                best_weights, prev_population = optimize_portfolio(
                    fhs_excess_returns, metric_func=ga_eval_func, warm_start_pop=prev_population, **kwargs
                )
                try:
                    best_val = ga_eval_func(fhs_excess_returns, best_weights)
                    is_finite = True
                except ValueError:
                    # Metric was undefined; fallback to mean excess return
                    port_excess = np.sum(fhs_excess_returns * best_weights, axis=1)
                    best_val = float(np.mean(port_excess))
                    is_finite = False

                current_weights = best_weights
            else:
                logger.warning(f"[{date_str}] Falling back to previous weights")
                # Keep previously aligned weights unchanged because fit failed and we can safely reuse old weights
                current_weights = cast(NDArray[np.float64], prev_full_weights[valid_cols].values)

            full_weights[valid_cols] = current_weights

            # Calculate turnover
            if is_first_day:
                turnover = 1.0
            else:
                # Evolve yesterday's target weights using actual observed returns on current_t
                prev_test_obs = df.iloc[current_t].fillna(0).values
                simple_obs = np.exp(prev_test_obs)
                adjusted_weights = prev_full_weights * simple_obs

                adj_sum = adjusted_weights.sum()
                if adj_sum > 0:
                    adjusted_weights /= adj_sum
                    turnover = float(np.sum(np.abs(full_weights - adjusted_weights)))
                else:
                    turnover = float("nan")

            prev_full_weights = full_weights.copy()
            prev_valid_assets = valid_assets_list

            # Portfolio return calculation
            simple_test_obs = np.exp(test_obs) - 1.0
            port_simple_ret = float(np.sum(current_weights * simple_test_obs))

        # ROW BUILDING PHASE
        row_res: dict[str, Any] = {
            "date": test_date,
            "log_likelihood": ll,
        }

        # Portfolio weights (fill with NaN if not optimized)
        for col in df.columns:
            row_res[f"weight_{col}"] = full_weights[col] if optimize_portfolio_flag else np.nan

        # Remaining core details
        row_res["portfolio_return"] = port_simple_ret
        row_res["turnover"] = turnover
        row_res["best_ga_metric_value"] = best_val
        row_res["ga_metric_is_finite"] = is_finite

        results_list.append(row_res)

    results_df = pd.DataFrame(results_list)
    results_df.set_index("date", inplace=True)

    # Save output to specified path as Parquet
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_parquet(output_path)

    return results_df
