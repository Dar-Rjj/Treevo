import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Volatility-normalized momentum using 3-day rolling standard deviation
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    volatility_5d = df['close'].pct_change().rolling(window=5).std()
    vol_normalized_momentum = momentum_3d / (volatility_5d + 1e-7)
    
    # Volume confirmation using volume-weighted price change
    volume_weighted_return = (df['close'].pct_change() * df['volume']).rolling(window=3).mean()
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_confirmation = volume_weighted_return * np.tanh(volume_trend / (df['volume'].rolling(window=5).std() + 1e-7))
    
    # Regime shift detection using price acceleration and volatility regime
    price_acceleration = df['close'].pct_change().diff().rolling(window=3).mean()
    volatility_regime = df['high'].rolling(window=5).std() / df['low'].rolling(window=5).std()
    regime_shift = np.tanh(price_acceleration * volatility_regime)
    
    # Adaptive bounds using rolling percentiles for dynamic scaling
    rolling_vol_norm = vol_normalized_momentum.rolling(window=10)
    adaptive_vol = rolling_vol_norm.apply(lambda x: x.iloc[-1] / (np.percentile(x, 75) - np.percentile(x, 25) + 1e-7))
    
    # Synergistic multiplicative combination with smooth transitions
    factor = (np.tanh(adaptive_vol) * 
              (1 + np.tanh(volume_confirmation)) * 
              (1 + regime_shift))
    
    return factor
