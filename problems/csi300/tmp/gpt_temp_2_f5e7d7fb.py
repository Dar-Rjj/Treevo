import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, M=20):
    # Calculate Log Returns
    close_prices = df['close']
    log_returns = np.log(close_prices).diff()

    # Compute Momentum
    cum_log_returns = log_returns.rolling(window=N).sum()
    
    # Define and apply dynamic thresholds
    upper_threshold = cum_log_returns.quantile(0.8)
    lower_threshold = cum_log_returns.quantile(0.2)
    clipped_log_returns = cum_log_returns.clip(lower=lower_threshold, upper=upper_threshold)

    # Adjust for Volume
    volumes = df['volume']
    volume_mean = volumes.rolling(window=N).mean()
    relative_volume = volumes / volume_mean
    volume_adjusted_momentum = clipped_log_returns * relative_volume

    # Determine Absolute Price Changes
    abs_price_changes = close_prices.diff().abs()

    # Calculate Advanced Volatility Measures
    std_abs_price_changes = abs_price_changes.rolling(window=M).std()
    ema_abs_price_changes = abs_price_changes.ewm(span=M, adjust=False).mean()
    iqr_abs_price_changes = abs_price_changes.rolling(window=M).quantile(0.75) - abs_price_changes.rolling(window=M).quantile(0.25)

    # Final Factor Calculation
    # Combine weighted components
    weights = np.array([0.4, 0.3, 0.3])
    factor = (volume_adjusted_momentum * weights[0] + 
              std_abs_price_changes * weights[1] + 
              ema_abs_price_changes * weights[2] +
              iqr_abs_price_changes * weights[3])

    # Ensure weights sum to 1
    factor = (factor / weights.sum())

    return factor

# Example usage
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
