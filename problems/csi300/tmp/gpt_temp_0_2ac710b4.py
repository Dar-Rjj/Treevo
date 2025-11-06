import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Momentum acceleration with volume breakout confirmation
    # Uses price acceleration (2nd derivative) with dynamic volume weighting
    
    # Price acceleration: difference between recent and longer-term momentum
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    momentum_acceleration = short_momentum - medium_momentum
    
    # Volatility normalization using robust rolling median absolute deviation
    price_range = df['high'] - df['low']
    volatility_mad = price_range.rolling(window=15).apply(lambda x: (x - x.median()).abs().median())
    
    # Volume breakout detection using percentile-based thresholds
    volume_roll = df['volume'].rolling(window=20)
    volume_median = volume_roll.median()
    volume_iqr = volume_roll.quantile(0.75) - volume_roll.quantile(0.25)
    volume_breakout = (df['volume'] - volume_median) / (volume_iqr + 1e-7)
    
    # Dynamic weighting: stronger volume breakouts get higher weight
    volume_weight = 1 + (volume_breakout.abs() * 0.5)
    
    # Combine components with clean interaction
    # Momentum acceleration normalized by robust volatility, amplified by volume breakout strength
    alpha_factor = (momentum_acceleration / (volatility_mad + 1e-7)) * volume_breakout * volume_weight
    
    return alpha_factor
