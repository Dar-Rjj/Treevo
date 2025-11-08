import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Short-Term Price Reversal
    # Calculate 1-day Lagged Return (t-1 to t-2)
    lagged_return_1d = (df['close'].shift(1) - df['close'].shift(2)) / df['close'].shift(2)
    
    # Calculate 3-day Rolling Return (t to t-3)
    rolling_return_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Compute Volatility Adjustment
    # Calculate 10-day Rolling Price Range
    rolling_high = df['high'].rolling(window=10, min_periods=1).max()
    rolling_low = df['low'].rolling(window=10, min_periods=1).min()
    price_range_10d = (rolling_high - rolling_low) / df['close'].shift(1)
    
    # Calculate Average True Range
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr_10d = true_range.rolling(window=10, min_periods=1).mean() / df['close'].shift(1)
    
    # Compute Liquidity Confirmation
    # Calculate Dollar Volume Trend
    dollar_volume = df['close'] * df['volume']
    
    def linear_slope(series):
        x = np.arange(len(series))
        if len(series) < 2:
            return 0
        return np.polyfit(x, series.values, 1)[0]
    
    dollar_volume_slope = dollar_volume.rolling(window=5, min_periods=2).apply(linear_slope, raw=False)
    
    # Calculate Bid-Ask Spread Proxy
    price_efficiency = (df['high'] - df['low']) / df['close']
    avg_price_efficiency = price_efficiency.rolling(window=5, min_periods=1).mean()
    
    # Combine Signals with Volatility Weighting
    # Avoid division by zero
    price_range_10d_safe = price_range_10d.replace(0, np.nan)
    atr_10d_safe = atr_10d.replace(0, np.nan)
    avg_price_efficiency_safe = avg_price_efficiency.replace(0, np.nan)
    
    component1 = lagged_return_1d / price_range_10d_safe
    component2 = rolling_return_3d / atr_10d_safe
    component3 = component1 * dollar_volume_slope
    component4 = component2 * (1 / avg_price_efficiency_safe)
    
    # Sum the four volatility-weighted components
    factor = component1 + component2 + component3 + component4
    
    return factor.fillna(0)
