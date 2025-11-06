import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor: Multi-Horizon Price-Volume Divergence with Volatility Regime Adjustment
    
    Economic intuition: Captures the divergence between price momentum and volume patterns across 
    multiple time horizons (5, 13, 34 days - Fibonacci sequence for natural market cycles), 
    adjusted for volatility regimes. The factor identifies stocks where price momentum is 
    accelerating faster than volume expansion (potential breakout) or decelerating slower than 
    volume contraction (potential reversal), while accounting for different volatility environments.
    
    Key innovations:
    - Price-volume divergence across Fibonacci time horizons (5, 13, 34 days)
    - Volatility regime classification using rolling percentiles
    - Multiplicative combination of divergence signals
    - No normalization, using raw ratios and multiplicative scaling
    - Simple, interpretable transformations with clear economic meaning
    """
    
    # Fibonacci time horizons for natural market cycles
    horizons = [5, 13, 34]
    
    # Price momentum components (simple returns)
    price_momentum = {}
    for horizon in horizons:
        price_momentum[horizon] = df['close'] / df['close'].shift(horizon) - 1
    
    # Volume momentum components (volume change ratios)
    volume_momentum = {}
    for horizon in horizons:
        volume_momentum[horizon] = df['volume'] / df['volume'].shift(horizon) - 1
    
    # Price-volume divergence ratios (momentum acceleration/deceleration)
    divergence_ratios = {}
    for horizon in horizons:
        # Ratio of price momentum to volume momentum
        # High ratio = price accelerating faster than volume
        # Low ratio = price decelerating slower than volume
        divergence_ratios[horizon] = (1 + price_momentum[horizon]) / (1 + volume_momentum[horizon])
    
    # Volatility regime using daily range (high-low relative to close)
    daily_volatility = (df['high'] - df['low']) / df['close']
    vol_regime_21d = daily_volatility.rolling(window=21).apply(
        lambda x: 2.0 if x.iloc[-1] > x.quantile(0.8) else  # High volatility
                 (0.5 if x.iloc[-1] < x.quantile(0.2) else 1.0)  # Low volatility / Normal
    )
    
    # Multi-horizon divergence combination (geometric mean)
    combined_divergence = (divergence_ratios[5] * divergence_ratios[13] * divergence_ratios[34]) ** (1/3)
    
    # Final factor: Combined divergence adjusted for volatility regime
    # In high volatility: Emphasize divergence signals (×2)
    # In low volatility: De-emphasize divergence signals (×0.5)
    # In normal volatility: Use raw divergence (×1)
    factor = combined_divergence * vol_regime_21d
    
    return factor
