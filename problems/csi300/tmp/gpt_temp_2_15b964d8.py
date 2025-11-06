import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price trend persistence (directional consistency)
    price_direction = np.sign(close.diff(3))
    trend_strength = (close.rolling(10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0))
    trend_persistence = price_direction * np.abs(trend_strength)
    
    # Volume entropy (regime detection)
    volume_rolling = volume.rolling(20)
    volume_quantiles = volume_rolling.rank(pct=True)
    volume_entropy = -volume_quantiles * np.log(volume_quantiles + 1e-8) - (1-volume_quantiles) * np.log(1-volume_quantiles + 1e-8)
    
    # Liquidity-adjusted momentum
    price_range = (high - low).rolling(5).mean()
    normalized_range = price_range / close.rolling(5).mean()
    liquidity_signal = trend_persistence * (1 - volume_entropy) * normalized_range
    
    # Regime-weighted factor
    volume_regime = volume.rolling(10).std() / volume.rolling(30).std()
    regime_weight = 1 / (1 + np.exp(-5 * (volume_regime - 1)))
    
    heuristics_matrix = liquidity_signal * regime_weight
    
    return heuristics_matrix
