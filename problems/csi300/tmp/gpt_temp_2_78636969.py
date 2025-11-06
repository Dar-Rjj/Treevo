import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Geometric decay-weighted momentum (5-day with exponential decay weights)
    decay_rate = 0.7
    momentum_weights = [decay_rate ** i for i in range(5)]
    momentum_weights = [w / sum(momentum_weights) for w in momentum_weights]  # Normalize weights
    
    weighted_momentum = sum(
        momentum_weights[i] * (df['close'] - df['close'].shift(i+1)) / df['close'].shift(i+1)
        for i in range(5)
    )
    
    # Volume-weighted momentum with decay
    volume_weights = df['volume'].rolling(window=5).apply(
        lambda x: np.prod([decay_rate ** i * x.iloc[i] for i in range(len(x))]) ** (1/len(x)),
        raw=False
    )
    volume_weighted_momentum = weighted_momentum * volume_weights
    
    # Volatility normalization using geometric mean of recent volatility
    vol_1d = (df['high'] - df['low']) / df['close']
    vol_5d_geo = vol_1d.rolling(window=5).apply(lambda x: np.prod(x) ** (1/len(x)), raw=False)
    
    # Volume flow with geometric decay
    volume_flow = df['volume'] / df['volume'].shift(1)
    volume_decay_geo = volume_flow.rolling(window=3).apply(
        lambda x: np.prod([decay_rate ** i * x.iloc[i] for i in range(len(x))]) ** (1/len(x)),
        raw=False
    )
    
    # Intraday geometric signals
    opening_gap_ratio = (df['open'] / df['close'].shift(1)) - 1
    intraday_range_utilization = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    
    # Combine components using geometric mean approach
    components = [
        np.sign(volume_weighted_momentum) * (np.abs(volume_weighted_momentum) ** 0.4),
        np.sign(volume_decay_geo) * (np.abs(volume_decay_geo) ** 0.3),
        np.sign(opening_gap_ratio) * (np.abs(opening_gap_ratio) ** 0.2),
        np.sign(intraday_range_utilization) * (np.abs(intraday_range_utilization) ** 0.1)
    ]
    
    # Geometric mean preserving signs
    factor = np.prod([np.abs(c) for c in components], axis=0) ** (1/len(components))
    factor *= np.prod([np.sign(c) for c in components], axis=0)
    
    # Normalize by recent geometric volatility for robustness
    volatility_normalizer = vol_5d_geo + 1e-7
    factor = factor / volatility_normalizer
    
    return factor
