import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor: Multi-Horizon Price-Volume-Range Geometric Alignment with True Range Normalization
    
    Economic intuition: Combines geometric alignment of price momentum, volume intensity, and range efficiency
    across three time horizons (5, 13, 34 days - Fibonacci sequence for natural market cycles). 
    Uses true range for volatility normalization and dollar volume for liquidity adjustment.
    The factor identifies stocks with strong, volume-confirmed momentum across multiple timeframes
    while exhibiting efficient price movement (small ranges relative to momentum).
    
    Key innovations:
    - Fibonacci time horizons (5, 13, 34) for natural market cycle alignment
    - Range efficiency component (momentum per unit of true range)
    - Dollar volume weighted smoothing for liquidity consideration
    - True range based volatility normalization
    - Pure geometric combinations without arithmetic operations
    """
    
    # Calculate true range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': (df['high'] - df['close'].shift(1)).abs(),
        'lc': (df['low'] - df['close'].shift(1)).abs()
    }).max(axis=1)
    
    # Dollar volume for liquidity adjustment
    dollar_volume = df['close'] * df['volume']
    
    # Multi-horizon components (5, 13, 34 days)
    horizons = [5, 13, 34]
    
    # Price momentum ratios (simple ratio instead of returns)
    momentum_components = []
    for horizon in horizons:
        momentum = df['close'] / df['close'].shift(horizon)
        # EMA smoothing with horizon-dependent span
        momentum_smooth = momentum.ewm(span=horizon).mean()
        momentum_components.append(momentum_smooth)
    
    # Volume intensity ratios
    volume_components = []
    for horizon in horizons:
        volume_intensity = df['volume'] / df['volume'].rolling(window=horizon).mean()
        volume_components.append(volume_intensity)
    
    # Range efficiency (momentum per unit of true range)
    range_components = []
    for horizon in horizons:
        momentum = df['close'] / df['close'].shift(horizon)
        avg_true_range = true_range.rolling(window=horizon).mean()
        range_efficiency = momentum / avg_true_range
        range_components.append(range_efficiency)
    
    # Geometric alignment across horizons for each component
    momentum_geo = (momentum_components[0] * momentum_components[1] * momentum_components[2]) ** (1/3)
    volume_geo = (volume_components[0] * volume_components[1] * volume_components[2]) ** (1/3)
    range_geo = (range_components[0] * range_components[1] * range_components[2]) ** (1/3)
    
    # Core factor: Geometric combination of all aligned components
    raw_factor = (momentum_geo * volume_geo * range_geo) ** (1/3)
    
    # Dollar volume weighted smoothing for liquidity robustness
    factor_weighted = raw_factor.rolling(window=21).apply(
        lambda x: np.average(x, weights=dollar_volume.loc[x.index]), raw=False
    )
    
    # True range based volatility normalization
    factor_volatility = factor_weighted.rolling(window=34).std()
    avg_true_range_34d = true_range.rolling(window=34).mean()
    
    # Final normalized factor
    factor = factor_weighted / (factor_volatility * avg_true_range_34d)
    
    return factor
