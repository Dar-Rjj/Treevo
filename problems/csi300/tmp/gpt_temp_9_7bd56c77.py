import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def heuristics_v2(df):
    # Parameters
    N = 20  # Number of days for log returns
    M = 60  # Number of days for volatility measures
    weights = {'momentum': 0.4, 'volume': 0.3, 'volatility': 0.3}

    # Step 1: Calculate Log Returns
    close_prices = df['close']
    log_returns = np.log(close_prices).diff(N)

    # Step 2: Compute Momentum
    cum_log_returns = log_returns.cumsum()
    upper_threshold = cum_log_returns.quantile(0.95)
    lower_threshold = cum_log_returns.quantile(0.05)
    momentum = np.clip(cum_log_returns, lower_threshold, upper_threshold)

    # Step 3: Adjust for Volume
    volumes = df['volume']
    mean_volume = volumes.rolling(window=M).mean()
    volume_adjusted_momentum = momentum * (volumes / mean_volume)

    # Step 4: Determine Absolute Price Changes
    abs_price_changes = close_prices.diff().abs()

    # Step 5: Calculate Advanced Volatility Measures
    std_dev = abs_price_changes.rolling(window=M).std()
    ema_abs_price_changes = abs_price_changes.ewm(span=M).mean()
    iqr = abs_price_changes.rolling(window=M).quantile(0.75) - abs_price_changes.rolling(window=M).quantile(0.25)
    kurt = abs_price_changes.rolling(window=M).apply(kurtosis, raw=False)
    skewness = abs_price_changes.rolling(window=M).apply(skew, raw=False)

    # Combine volatility measures
    volatility_measures = (std_dev + ema_abs_price_changes + iqr + kurt + skewness) / 5

    # Step 6: Final Factor Calculation
    combined_factor = (
        weights['momentum'] * volume_adjusted_momentum +
        weights['volume'] * (volumes / mean_volume) +
        weights['volatility'] * volatility_measures
    )

    return combined_factor
