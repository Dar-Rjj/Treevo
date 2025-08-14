import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=10, M=20, short_ema_period=10, long_ema_period=50):
    # Obtain Close Prices
    close_prices = df['close']
    
    # Compute Log Return over N Days
    log_returns = np.log(close_prices).diff(N)
    
    # Retrieve Volumes
    volumes = df['volume']
    
    # Incorporate Volume into Momentum
    avg_volume = volumes.rolling(window=N).mean()
    vol_adjusted_momentum = log_returns * avg_volume
    
    # Determine Absolute Price Changes
    abs_price_changes = close_prices.diff().abs()
    std_dev_price_changes = abs_price_changes.rolling(window=M).std()
    
    # Compute Average True Range (ATR)
    high_low_diff = df['high'] - df['low']
    high_close_diff = (df['high'] - df['close'].shift()).abs()
    low_close_diff = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low_diff, high_close_diff, low_close_diff], axis=1).max(axis=1)
    atr = tr.rolling(window=N).mean()
    
    # Combine ATR and Standard Deviation
    combined_volatility = 0.5 * atr + 0.5 * std_dev_price_changes
    
    # Use Exponential Moving Average (EMA) of Close prices
    ema_short = close_prices.ewm(span=short_ema_period, adjust=False).mean()
    ema_long = close_prices.ewm(span=long_ema_period, adjust=False).mean()
    ema_diff = ema_short - ema_long
    
    # Apply Weighting Scheme to Recent Observations
    weights = np.exp(-np.arange(N)[::-1] / N)
    weighted_momentum = np.convolve(vol_adjusted_momentum, weights, mode='valid')[:len(df)]
    weighted_volatility = np.convolve(combined_volatility, weights, mode='valid')[:len(df)]
    
    # Final Factor Calculation
    factor = weighted_momentum + weighted_volatility + ema_diff
    return pd.Series(factor, index=df.index, name='heuristics_v2_factor')

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=True, index_col=0)
# factor_values = heuristics_v2(df)
