import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Simplified Regime-Adaptive Momentum-Volume Factor
    Combines momentum acceleration with volume efficiency in a volatility-aware framework
    Positive values indicate accelerating momentum with efficient volume in stable regimes
    Negative values suggest momentum decay or inefficient volume in volatile conditions
    """
    # Momentum acceleration across geometric timeframes
    momentum_1d = df['close'] / df['close'].shift(1) - 1
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_6d = df['close'] / df['close'].shift(6) - 1
    
    # Volatility-based regime detection
    volatility = df['close'].pct_change().rolling(window=10).std()
    vol_regime = volatility.rolling(window=20).apply(lambda x: (x > x.median()).astype(float).iloc[-1])
    
    # Momentum acceleration components
    accel_short = momentum_1d - momentum_3d
    accel_medium = momentum_3d - momentum_6d
    
    # Volume efficiency: price impact per unit volume
    price_range = (df['high'] - df['low']) / df['close']
    volume_efficiency = price_range / (df['volume'] + 1e-8)
    
    # Volume trend consistency
    volume_trend = df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Regime-adaptive blending
    # Low volatility: emphasize momentum acceleration
    # High volatility: emphasize volume efficiency
    momentum_component = (accel_short * accel_medium) * (1 - vol_regime)
    volume_component = (volume_efficiency * volume_trend) * vol_regime
    
    alpha_factor = momentum_component + volume_component
    
    return alpha_factor
