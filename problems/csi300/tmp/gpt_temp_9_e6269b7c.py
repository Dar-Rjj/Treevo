import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Novel alpha factor: Multi-timeframe momentum convergence with volume regime detection
    # Combines momentum across different time horizons with volume regime identification
    
    # Triple timeframe momentum convergence (3-day, 5-day, 8-day)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    mom_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    # Momentum convergence score (agreement across timeframes)
    momentum_convergence = (mom_3d * mom_5d * mom_8d) ** (1/3)
    
    # Volume regime detection using multi-period comparisons
    volume_short = df['volume'].rolling(window=3).mean()
    volume_medium = df['volume'].rolling(window=8).mean()
    volume_long = df['volume'].rolling(window=15).mean()
    
    # Volume regime strength (short-term acceleration vs medium-term trend)
    volume_regime = (volume_short / volume_medium) * (volume_medium / volume_long)
    
    # Price efficiency using intraday range utilization
    daily_range = df['high'] - df['low']
    close_position = (df['close'] - df['low']) / (daily_range + 1e-7)
    range_efficiency = close_position.rolling(window=5).mean()
    
    # Adaptive volatility using rolling range percentiles
    range_percentile = daily_range.rolling(window=10).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-7)
    )
    
    # Combine components with regime-dependent weighting
    raw_alpha = momentum_convergence * volume_regime * range_efficiency
    
    # Volatility-adjusted factor with regime sensitivity
    alpha_factor = raw_alpha / (abs(range_percentile) + 1e-7)
    
    return alpha_factor
