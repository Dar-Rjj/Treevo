import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Exponential Moving Average (EMA) of High-Low Range with adaptive lookback period
    def adaptive_lookback_volatility(df, window=20):
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(window=window).std()
        return volatility
    
    volatility = adaptive_lookback_volatility(df)
    ema_high_low_range = high_low_range.ewm(span=volatility.rolling(window=20).mean() * 20, adjust=False).mean()
    
    # Calculate Volume Adjusted High-Low Range
    volume_adjusted_high_low = df['volume'] * high_low_range
    
    # Calculate EMA of Volume Adjusted High-Low Range with adaptive lookback period
    ema_volume_adjusted_high_low = volume_adjusted_high_low.ewm(span=volatility.rolling(window=20).mean() * 20, adjust=False).mean()
    
    # Calculate Momentum Factor
    momentum_factor = ema_volume_adjusted_high_low - ema_high_low_range
    
    # Integrate Adaptive Volatility
    daily_returns = df['close'].pct_change().dropna()
    std_daily_returns = daily_returns.rolling(window=20).std()
    
    # Adjust Momentum Factor by Volatility
    adjusted_momentum_factor = momentum_factor / std_daily_returns
    
    return adjusted_momentum_factor.dropna()
