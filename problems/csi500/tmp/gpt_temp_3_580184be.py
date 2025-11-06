import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum blend with regime-aware volatility and persistent volume alignment
    # Combines short-term (2-day) and medium-term (5-day) momentum signals
    # Uses adaptive volatility based on price range regimes
    # Aligns with persistent volume trends for signal confirmation
    
    # Multi-timeframe momentum blend
    momentum_short = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    momentum_medium = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_blend = momentum_short * 0.6 + momentum_medium * 0.4
    
    # Adaptive volatility using regime-aware true range
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Volatility regime detection using rolling percentiles
    vol_regime = true_range.rolling(window=10).apply(lambda x: (x.iloc[-1] > x.quantile(0.7)) * 1 + (x.iloc[-1] < x.quantile(0.3)) * -1, raw=False)
    regime_adjusted_vol = true_range * (1 + vol_regime * 0.2)
    
    # Persistent volume trend alignment
    volume_persistence = (df['volume'].rolling(window=3).mean() > df['volume'].rolling(window=10).mean()).astype(int)
    volume_trend_strength = df['volume'].rolling(window=3).mean() / df['volume'].rolling(window=10).mean()
    volume_alignment = volume_persistence * volume_trend_strength
    
    # Amount-based trend confirmation
    amount_trend = (df['amount'].rolling(window=3).mean() - df['amount'].rolling(window=10).mean()) / df['amount'].rolling(window=10).mean()
    
    # Combined factor: momentum blend normalized by regime-aware volatility,
    # multiplied by volume alignment and amount trend for confirmation
    alpha_factor = (momentum_blend / (regime_adjusted_vol + 1e-7)) * volume_alignment * (1 + amount_trend)
    
    return alpha_factor
