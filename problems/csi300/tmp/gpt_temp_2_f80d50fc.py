import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel alpha factor: Multi-timeframe weighted momentum blended with volume-to-median ratios and volatility dampening
    # Economic rationale: Combines short-term (3-5 day) momentum signals with volume activity relative to recent median,
    # while using rolling volatility to filter noise and adaptive windows for robustness
    
    # Adaptive window sizes based on recent volatility regime
    vol_10d = df['close'].pct_change().rolling(10).std()
    adaptive_window = np.where(vol_10d > vol_10d.rolling(20).median(), 3, 5)
    
    # Multi-timeframe weighted momentum (3-5 days with decaying weights)
    momentum_weights = []
    for i, window in enumerate(adaptive_window):
        weights = np.array([0.5, 0.3, 0.2]) if window == 3 else np.array([0.4, 0.3, 0.2, 0.1])
        momentum = 0
        for j, weight in enumerate(weights):
            momentum += weight * (df['close'].iloc[i] / df['close'].shift(j+1).iloc[i] - 1)
        momentum_weights.append(momentum)
    
    weighted_momentum = pd.Series(momentum_weights, index=df.index)
    
    # Volume-to-median ratio with 10-day lookback
    volume_median_10d = df['volume'].rolling(10).median()
    volume_ratio = df['volume'] / (volume_median_10d + 1e-7)
    
    # Rolling volatility dampening (15-day standard deviation of returns)
    returns = df['close'].pct_change()
    rolling_vol = returns.rolling(15).std()
    
    # Combine components: momentum amplified by volume activity, dampened by volatility
    factor = weighted_momentum * volume_ratio / (rolling_vol + 1e-7)
    
    return factor
