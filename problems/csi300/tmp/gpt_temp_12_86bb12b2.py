import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Dynamic volume-normalized volatility momentum factor
    # Combines adaptive momentum windows with volume-scaled volatility measures
    
    # Dynamic momentum calculation based on recent volatility regime
    # Use shorter window (3-day) during high volatility, longer (7-day) during low volatility
    vol_regime = (df['high'] - df['low']).rolling(window=10).std()
    high_vol_threshold = vol_regime.rolling(window=20).quantile(0.7)
    momentum_window = np.where(vol_regime > high_vol_threshold, 3, 7)
    
    # Calculate momentum with dynamic window
    momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(momentum_window):
            window = int(momentum_window[i])
            momentum.iloc[i] = df['close'].iloc[i] / df['close'].iloc[i - window] - 1
    
    # Volume-normalized true range volatility
    volume_weighted_tr = ((df['high'] - df['low']) * df['volume']).rolling(window=10).mean()
    volume_baseline = df['volume'].rolling(window=20).mean()
    normalized_volatility = volume_weighted_tr / (volume_baseline + 1e-7)
    
    # Volume momentum with adaptive confirmation
    volume_momentum_window = np.where(vol_regime > high_vol_threshold, 5, 10)
    volume_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= max(volume_momentum_window):
            window = int(volume_momentum_window[i])
            volume_momentum.iloc[i] = df['volume'].iloc[i] / df['volume'].iloc[i - window] - 1
    
    # Combine factors: momentum adjusted by volume-normalized volatility and volume confirmation
    alpha = momentum * (1 + volume_momentum) / (normalized_volatility + 1e-7)
    
    return alpha
