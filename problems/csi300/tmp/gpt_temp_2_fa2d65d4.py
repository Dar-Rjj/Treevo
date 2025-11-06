import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel alpha factor combining momentum acceleration, volume divergence, and volatility-normalized extremes
    # Captures stocks with accelerating momentum confirmed by volume divergence, normalized by smoothed volatility
    
    # Exponential smoothing parameters
    alpha_fast = 0.3
    alpha_slow = 0.1
    
    # 1. Momentum acceleration: Difference between fast and slow EMA momentum
    fast_momentum = (df['close'] - df['close'].ewm(alpha=alpha_fast).mean()) / df['close'].ewm(alpha=alpha_fast).mean()
    slow_momentum = (df['close'] - df['close'].ewm(alpha=alpha_slow).mean()) / df['close'].ewm(alpha=alpha_slow).mean()
    momentum_acceleration = fast_momentum - slow_momentum
    
    # 2. Volume divergence: Current volume vs exponentially smoothed volume with alignment check
    volume_ema = df['volume'].ewm(alpha=alpha_fast).mean()
    volume_divergence = df['volume'] / (volume_ema + 1e-7)
    
    # Signal alignment: Only use volume divergence when it confirms momentum direction
    volume_confirmation = volume_divergence * np.sign(momentum_acceleration)
    
    # 3. Volatility normalization using ATR (Average True Range) with exponential smoothing
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    atr = tr.ewm(alpha=alpha_slow).mean()
    
    # 4. Extreme divergence detection using percentile ranking
    momentum_rank = momentum_acceleration.rolling(20).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 1.0, raw=False)
    volume_rank = volume_confirmation.rolling(20).apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) * 1.0, raw=False)
    extreme_divergence = momentum_rank * volume_rank
    
    # Combine components multiplicatively with volatility normalization and extreme divergence emphasis
    factor = (momentum_acceleration * volume_confirmation * extreme_divergence) / (atr + 1e-7)
    
    return factor
