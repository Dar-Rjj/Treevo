import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(data):
    # Calculate Intraday Return
    intraday_return = data['close'] - data['open']
    
    # Calculate Intraday High-Low Range
    high_low_range = data['high'] - data['low']
    
    # Combine Intraday Return and High-Low Range
    combined_factor = intraday_return * high_low_range
    
    # Adaptive Exponential Moving Average (AEMA) period based on recent volatility
    def adaptive_ema(x, vol_lookback=20):
        std_dev = x.rolling(window=vol_lookback).std()
        period = 2 / (1 + np.exp(-zscore(std_dev)))  # Sigmoid function to adjust EMA period
        ema = x.ewm(span=period, adjust=False).mean()
        return ema
    
    smoothed_factor = adaptive_ema(combined_factor)
    
    # Apply Volume Weighting
    volume_weighted_factor = smoothed_factor * data['volume']
    
    # Incorporate Previous Day's Closing Gap
    prev_day_close = data['close'].shift(1)
    closing_gap = data['open'] - prev_day_close
    gap_adjusted_factor = volume_weighted_factor + closing_gap
    
    # Integrate Long-Term Momentum
    long_term_return = data['close'] - data['close'].shift(50)
    adjusted_long_term_return = long_term_return / high_low_range
    
    # Include Dynamic Volatility Component
    rolling_std = intraday_return.rolling(window=20).std()
    
    # Calculate Average True Range (ATR)
    true_range = pd.DataFrame({
        'h-l': data['high'] - data['low'],
        'h-pc': abs(data['high'] - data['close'].shift(1)),
        'l-pc': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    atr = true_range.rolling(window=14).mean()
    
    # Adjust Volatility Component with Volume
    volatility_component = (rolling_std + atr) * data['volume']
    
    # Final Factor Calculation
    final_factor = gap_adjusted_factor + adjusted_long_term_return + volatility_component
    
    # Apply Non-Linear Transformation
    final_factor = np.log(1 + final_factor)
    
    return final_factor
