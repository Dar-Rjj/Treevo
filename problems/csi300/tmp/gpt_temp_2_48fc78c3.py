import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-horizon momentum components (1-day, 3-day, 5-day)
    momentum_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volatility-normalized volume (5-day rolling volatility)
    vol_5d = df['close'].pct_change().rolling(window=5).std()
    volume_norm = df['volume'] / (vol_5d + 1e-7)
    volume_momentum = (volume_norm - volume_norm.shift(1)) / (volume_norm.shift(1) + 1e-7)
    
    # Regime-smooth interactions using rolling averages
    regime_short = df['close'].rolling(window=5).mean()
    regime_long = df['close'].rolling(window=20).mean()
    regime_smooth = (regime_short - regime_long) / regime_long
    
    # Price efficiency components
    daily_range = df['high'] - df['low']
    range_efficiency = (df['close'] - df['low']) / (daily_range + 1e-7)
    
    # Adaptive scaling using rolling percentiles for bounded values
    momentum_1d_scaled = momentum_1d.rolling(window=20).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    momentum_3d_scaled = momentum_3d.rolling(window=20).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() > 0 else 0)
    
    # Additive components for clarity
    additive_base = (momentum_1d_scaled + momentum_3d_scaled + momentum_5d + 
                    volume_momentum + regime_smooth + range_efficiency)
    
    # Multiplicative synergy terms
    momentum_synergy = momentum_1d * momentum_3d * momentum_5d
    regime_volume_synergy = regime_smooth * volume_momentum
    
    # Final factor with economic rationale: blend additive clarity with multiplicative synergy
    factor = additive_base * (1 + momentum_synergy) * (1 + regime_volume_synergy)
    
    return factor
