import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor: Multi-frequency Volume Acceleration with Volatility-Regime Adaptive Scaling
    
    This factor enhances volume acceleration analysis by incorporating multiple frequency bands
    and adaptive volatility scaling that responds to different market regimes.
    
    Interpretation:
    - High positive values: Strong multi-frequency volume acceleration in low-volatility regimes
    - High negative values: Strong selling pressure across multiple timeframes in stable conditions
    - Values near zero: Weak volume signals, high volatility, or conflicting frequency patterns
    
    Key innovations:
    - Multi-frequency volume acceleration (1, 3, 5-day horizons)
    - Volatility regime detection using rolling percentiles
    - Adaptive scaling that penalizes high-volatility periods more aggressively
    - Amount-based institutional flow confirmation across frequencies
    - Signal convergence scoring for robust factor construction
    """
    
    # Multi-frequency volume acceleration
    volume_accel_1d = (df['volume'] - df['volume'].shift(1)) / (df['volume'].shift(1) + 1e-7)
    volume_accel_3d = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-7)
    volume_accel_5d = (df['volume'] - df['volume'].shift(5)) / (df['volume'].shift(5) + 1e-7)
    
    # Volume acceleration convergence (weighted by recency)
    volume_convergence = (3 * volume_accel_1d + 2 * volume_accel_3d + volume_accel_5d) / 6
    
    # Multi-frequency amount per share (institutional proxy)
    amount_per_share = df['amount'] / (df['volume'] + 1e-7)
    amount_accel_1d = (amount_per_share - amount_per_share.shift(1)) / (amount_per_share.shift(1) + 1e-7)
    amount_accel_3d = (amount_per_share - amount_per_share.shift(3)) / (amount_per_share.shift(3) + 1e-7)
    
    # Institutional flow confirmation
    institutional_flow = (2 * amount_accel_1d + amount_accel_3d) / 3
    
    # Volatility regime detection and adaptive scaling
    daily_range = (df['high'] - df['low']) / df['close']
    vol_regime = daily_range.rolling(window=20, min_periods=10).apply(
        lambda x: 2.0 if x.iloc[-1] > x.quantile(0.8) else 1.0 if x.iloc[-1] > x.quantile(0.6) else 0.5
    )
    
    # Price momentum for directional context (absolute to focus on signal strength)
    price_momentum = abs((df['close'] - df['close'].shift(1)) / df['close'].shift(1))
    
    # Combined factor: volume convergence × institutional flow × momentum, with adaptive volatility scaling
    alpha_factor = -price_momentum * volume_convergence * institutional_flow / (vol_regime + 1e-7)
    
    return alpha_factor
