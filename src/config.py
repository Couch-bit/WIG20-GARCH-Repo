# It's good to have the boundaries of the data periods globally in this file to avoid inconsistencies
VAL_START = "2018-01-01"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"
TEST_END = "2023-12-31"
TUNE_START = "2019-01-01"
TUNE_END = "2019-12-31"

# Window size can be changed here, but 500 is typical for GARCH
WINDOW_SIZE = 500

# Total tokens in optimization can also be tuned here
TOTAL_TOKENS = 1000

# Which models should be tested in the evaluation
EVAL_MEAN_MODELS = ["naive", "ar", "var", "var_lasso"]
EVAL_VOLATILITY_MODELS = ["naive", "ccc", "dcc", "go_garch"]

# Model combinations for test data and GA tuning in the format (<mean_model>, <volatility_model>, <display_name>)
TEST_MODELS = [
    ("naive", "naive", "Brak modeli"),
    ("var_lasso", "naive", "Tylko model średniej"),
    ("var_lasso", "ccc", "CCC"),
    ("var_lasso", "dcc", "DCC"),
    ("var_lasso", "go_garch", "GO-GARCH"),
]
