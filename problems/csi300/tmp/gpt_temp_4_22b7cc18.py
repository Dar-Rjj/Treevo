import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Volatility-Regime Adaptive Momentum-Volume Geometric Factor
    
    Economic intuition: Captures the geometric convergence of price momentum and volume 
    dynamics across asymmetric time horizons, with volatility-regime adaptive processing
    that enhances signal stability. The factor identifies stocks where momentum and volume
    exhibit consistent geometric alignment across different market cycles, with adaptive
    smoothing that responds to changing volatility conditions for robust signal generation.
    
    Key innovations:
    - Clean geometric combination of momentum and volume across three asymmetric horizons
    - Volatility-regime adaptive smoothing using price range characteristics
    - Multiplicative interaction between momentum strength and volume dynamics
    - Geometric mean across horizons for robust signal extraction
    - Regime-aware exponential smoothing for enhanced stability
    """
    
    # Asymmetric horizons for multi-timeframe geometric analysis
    horizons = [3, 8, 21]  # Short, medium, long-term market cycles
    
    # Initialize geometric components list
    geometric_components = []
    
    for horizon in horizons:
        # Geometric price momentum: normalized return over horizon
        price_momentum = (df['close'] / df['close'].shift(horizon)) ** (1/horizon) - 1
        
        # Geometric volume dynamics: volume relative to rolling geometric mean
        volume_geomean = df['volume'].rolling(window=horizon).apply(lambda x: x.prod() ** (1/len(x)))
        volume_dynamics = (df['volume'] / volume_geomean) ** (1/horizon) - 1
        
        # Geometric combination: momentum enhanced by volume dynamics
        momentum_volume_geo = price_momentum * (1 + volume_dynamics)
        
        geometric_components.append(momentum_volume_geo)
    
    # Pure geometric mean across all horizons
    geo_factor = (geometric_components[0] * geometric_components[1] * geometric_components[2]) ** (1/3)
    
    # Volatility regime detection using geometric range characteristics
    daily_range = (df['high'] - df['low']) / df['close']
    range_geovol = daily_range.rolling(window=13).apply(lambda x: x.prod() ** (1/len(x)))
    volatility_regime = (range_geovol / range_geovol.rolling(window=34).apply(lambda x: x.prod() ** (1/len(x)))).clip(lower=0.3, upper=3.0)
    
    # Volatility-regime adaptive geometric smoothing
    fast_geo_smooth = geo_factor.ewm(span=2).mean()
    slow_geo_smooth = geo_factor.ewm(span=13).mean()
    
    # Geometric interpolation based on volatility regime
    regime_weight = (volatility_regime - 0.3) / (3.0 - 0.3)  # Normalize to [0,1]
    regime_adjusted = (fast_geo_smooth ** (1 - regime_weight)) * (slow_geo_smooth ** regime_weight)
    
    # Final geometric refinement with multi-period smoothing
    smooth_5d = regime_adjusted.ewm(span=5).mean()
    smooth_8d = regime_adjusted.ewm(span=8).mean()
    
    final_factor = (smooth_5d * smooth_8d) ** 0.5
    
    return final_factor
