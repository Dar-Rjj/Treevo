import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=20, N=10, M=5):
    # Calculate Price Momentum
    close_momentum = df['close'].pct_change(n).fillna(0)
    open_momentum = df['open'].pct_change(n).fillna(0)
    
    # Calculate Enhanced Volume Adjusted Volatility
    high_low_range = df['high'] - df['low']
    volatility = high_low_range.rolling(window=n).std().fillna(0)
    volume_adjusted_volatility = volatility / df['volume']

    # Calculate Volume Surge Indicator
    volume_change_n_days = df['volume'] - df['volume'].shift(N).fillna(0)
    volume_change_m_days = df['volume'] - df['volume'].shift(M).fillna(0)
    volume_surge = (volume_change_n_days + volume_change_m_days).fillna(0)

    # Long-Term Volume-Weighted Average Return (Momentum Component)
    daily_returns_100 = df['close'].pct_change().fillna(0).rolling(window=100).sum()
    volume_weighted_return_100 = (daily_returns_100 * df['volume']).sum() / df['volume'].rolling(window=100).sum()

    # Short-Term Volume-Weighted Average Return (Reversal Component)
    daily_returns_5 = df['close'].pct_change().fillna(0).rolling(window=5).sum()
    volume_weighted_return_5 = (daily_returns_5 * df['volume']).sum() / df['volume'].rolling(window=5).sum()

    # Integrate High-Low Range into Volume-Weighted Returns
    high_low_range_current = df['high'] - df['low']
    volume_weighted_high_low_range = high_low_range_current * df['volume'] / df['volume'].rolling(window=5).sum()

    # Combine Factors
    price_momentum = (close_momentum + open_momentum) / 2
    adjusted_volatility = volume_adjusted_volatility
    adjusted_momentum = price_momentum * adjusted_volatility
    if volume_surge < 0:
        adjusted_momentum *= 0.8
    combined_momentum_reversal = volume_weighted_return_100 - volume_weighted_return_5 + volume_weighted_high_low_range
    final_alpha_factor = combined_momentum_reversal * adjusted_momentum

    return final_alpha_factor
