import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Detection
    daily_range = (df['high'] - df['low']) / df['close']
    vol_proxy = daily_range.rolling(window=10, min_periods=10).std()
    
    # Regime Classification
    vol_percentile = vol_proxy.rolling(window=30, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    high_vol_regime = (vol_percentile > 0.8).astype(float)
    low_vol_regime = (vol_percentile < 0.2).astype(float)
    medium_vol_regime = ((vol_percentile >= 0.2) & (vol_percentile <= 0.8)).astype(float)
    
    # Regime-Adaptive Momentum Signal
    short_return = df['close'] / df['close'].shift(3) - 1
    medium_return = df['close'] / df['close'].shift(8) - 1
    long_return = df['close'] / df['close'].shift(15) - 1
    
    momentum_score = np.sign(short_return * medium_return * long_return) * (
        np.abs(short_return * medium_return * long_return) ** (1/3)
    )
    
    regime_momentum = (
        momentum_score * high_vol_regime * 0.7 +
        momentum_score * medium_vol_regime * 1.0 +
        momentum_score * low_vol_regime * 1.3
    )
    
    # Volume-Price Convergence Signal
    recent_price_change = df['close'] / df['close'].shift(2) - 1
    previous_price_change = df['close'].shift(2) / df['close'].shift(4) - 1
    price_acceleration = recent_price_change - previous_price_change
    
    recent_volume_change = df['volume'] / df['volume'].shift(2) - 1
    previous_volume_change = df['volume'].shift(2) / df['volume'].shift(4) - 1
    volume_acceleration = recent_volume_change - previous_volume_change
    
    raw_convergence = price_acceleration * volume_acceleration
    convergence_score = np.tanh(raw_convergence)
    
    # Integrated Factor Construction
    raw_factor = regime_momentum * convergence_score
    
    daily_returns = df['close'].pct_change()
    recent_volatility = daily_returns.rolling(window=5, min_periods=5).std()
    
    scaled_factor = raw_factor / (recent_volatility + 0.0001)
    
    final_factor = np.tanh(scaled_factor)
    
    return final_factor
