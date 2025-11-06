import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive Momentum-Volume Synchronization with Dynamic Volatility Scaling
    
    Economic intuition: Captures the synchronization between price momentum and volume dynamics
    across adaptive time horizons, while dynamically adjusting for volatility regimes. The factor
    identifies stocks where price movements are consistently supported by volume patterns across
    multiple frequencies, suggesting sustainable directional moves. Dynamic volatility scaling
    adapts to changing market conditions, while volume-based weighting emphasizes stocks with
    meaningful trading activity.
    
    Key innovations:
    - Adaptive horizon selection (8, 13, 34 days) representing Fibonacci-based market cycles
    - Component synchronization via geometric mean of smoothed momentum-volume products
    - Dynamic volatility scaling using rolling volatility percentiles
    - Volume-based economic weighting for implementation practicality
    - Multiplicative combination preserving economic interpretability
    """
    
    # Adaptive horizons based on Fibonacci sequence for market cycle alignment
    horizons = [8, 13, 34]
    
    # Calculate momentum components with exponential smoothing
    momentum_components = []
    for horizon in horizons:
        momentum = (df['close'] / df['close'].shift(horizon) - 1)
        smoothed_momentum = momentum.ewm(span=horizon).mean()
        momentum_components.append(smoothed_momentum)
    
    # Calculate volume expansion components with smoothing
    volume_components = []
    for horizon in horizons:
        volume_ratio = df['volume'] / df['volume'].rolling(window=horizon).mean()
        smoothed_volume = volume_ratio.ewm(span=horizon).mean()
        volume_components.append(smoothed_volume)
    
    # Calculate volatility components using true range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    
    volatility_components = []
    for horizon in horizons:
        vol_ratio = true_range / true_range.rolling(window=horizon).mean()
        smoothed_volatility = vol_ratio.ewm(span=horizon).mean()
        volatility_components.append(smoothed_volatility)
    
    # Geometric combination of momentum-volume synchronization
    sync_components = []
    for i, horizon in enumerate(horizons):
        momentum_vol_sync = momentum_components[i] * volume_components[i]
        smoothed_sync = momentum_vol_sync.ewm(span=horizon).mean()
        sync_components.append(smoothed_sync)
    
    # Multiplicative combination of synchronized components
    raw_factor = pd.concat(sync_components, axis=1).apply(lambda x: x.prod() ** (1/len(horizons)), axis=1)
    
    # Dynamic volatility scaling using rolling percentile
    volatility_scale = true_range.rolling(window=34).apply(lambda x: x.rank(pct=True).iloc[-1])
    normalized_factor = raw_factor / (volatility_scale + 1e-7)
    
    # Volume-based economic weighting
    volume_weight = df['volume'].ewm(span=21).mean()
    final_factor = normalized_factor * volume_weight
    
    return final_factor
