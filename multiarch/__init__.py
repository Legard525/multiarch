"""multiarch: fast multivariate GARCH (CCC / DCC / aDCC / copula-DCC /
GO-GARCH) for Python."""

from .dcc import DCC, DCCResult
from .diagnostics import constant_correlation_test, hosking_test, var_backtest
from .ewma import ewma_covariance
from .gogarch import GOGARCH, GOGARCHResult
from .rolling import RollingForecast, rolling_forecast
from .simulate import simulate_dcc
from .univariate import UnivariateSpec, fit_univariate

__all__ = [
    "DCC", "DCCResult", "GOGARCH", "GOGARCHResult", "RollingForecast",
    "UnivariateSpec", "constant_correlation_test", "ewma_covariance",
    "fit_univariate", "hosking_test", "rolling_forecast", "simulate_dcc",
    "var_backtest",
]
__version__ = "0.2.0"
