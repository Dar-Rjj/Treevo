import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-Horizon Price-Volume Efficiency Factor with Volatility-Adjusted Momentum
    
    Economic intuition: Captures the efficiency of price movement relative to volume across 
    multiple time horizons (5, 13, 34 days - Fibonacci sequence for natural market cycles).
    The factor identifies stocks where price momentum is efficiently supported by volume 
    across different market cycles, adjusted for volatility regime changes.
    
    Key innovations:
    - Price-volume efficiency ratio across Fibonacci time horizons
    - Volatility regime adjustment using rolling volatility percentiles
    - Multiplicative combination of horizon-specific efficiencies
    - No normalization, maintaining raw economic relationships
    """
    
    # Fibonacci time horizons for natural market cycles
    horizons = [5, 13, 34]
    
    # Calculate price momentum across horizons
    price_momentum = {}
    for horizon in horizons:
        price_momentum[horizon] = (df['close'] / df['close'].shift(horizon) - 1)
    
    # Calculate volume efficiency (volume relative to price movement)
    volume_efficiency = {}
    for horizon in horizons:
        # Volume intensity relative to recent average
        vol_intensity = df['volume'] / df['volume'].rolling(window=horizon).mean()
        # Price-volume efficiency: momentum per unit of volume intensity
        volume_efficiency[horizon] = price_momentum[horizon] * vol_intensity
    
    # Calculate volatility regime using daily range
    daily_volatility = (df['high'] - df['low']) / df['close']
    vol_regime = {}
    for horizon in horizons:
        # Volatility percentile relative to recent history
        vol_regime[horizon] = daily_volatility.rolling(window=horizon).apply(
            lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
        )
    
    # Combine components multiplicatively for each horizon
    horizon_factors = {}
    for horizon in horizons:
        # Core factor: Price-volume efficiency adjusted for volatility regime
        # Negative vol_regime (low volatility) enhances positive signals
        horizon_factors[horizon] = volume_efficiency[horizon] * (1 - vol_regime[horizon])
    
    # Geometric mean across horizons for robust multi-timeframe signal
    combined_factor = 1.0
    for horizon in horizons:
        combined_factor *= horizon_factors[horizon]
    combined_factor = combined_factor ** (1/len(horizons))
    
    return combined_factor
