import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical patterns:
    - Multi-timeframe momentum divergence
    - Volume-weighted extreme reversal
    - Volatility-scaled efficiency momentum
    - Amount flow persistence patterns
    - Range-volume regime interaction
    """
    # Extract price and volume data
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Multi-Timeframe Momentum Divergence
    momentum_5d = close / close.shift(5) - 1
    momentum_10d = close / close.shift(10) - 1
    momentum_acceleration = momentum_5d / momentum_10d.replace(0, np.nan)
    
    volume_trend_ratio = volume / volume.shift(5)
    volume_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            volume_persistence.iloc[i] = ((volume.iloc[i-3:i+1] > volume.shift(1).iloc[i-3:i+1]).sum())
    
    bullish_divergence = momentum_5d * (1 - volume_trend_ratio)
    bearish_divergence = -momentum_5d * volume_trend_ratio
    multi_timeframe_divergence = momentum_acceleration * volume_persistence
    
    # Volume-Weighted Extreme Reversal
    price_extremity = pd.Series(index=df.index, dtype=float)
    volume_extremity = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 5:
            price_extremity.iloc[i] = (close.iloc[i] - low.iloc[i-5:i+1].min()) / (high.iloc[i-5:i+1].max() - low.iloc[i-5:i+1].min())
        if i >= 10:
            volume_extremity.iloc[i] = volume.iloc[i] / volume.iloc[i-10:i+1].median()
    
    combined_extreme = price_extremity * volume_extremity
    reversal_1d = -np.sign(close - close.shift(1)) * np.abs(close / close.shift(1) - 1)
    volume_weighted_reversal = reversal_1d * volume_extremity
    
    volatility_context = (high - low) / close
    volume_persistence_3d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            median_vol = volume.iloc[i-10:i+1].median() if i >= 10 else volume.iloc[:i+1].median()
            volume_persistence_3d.iloc[i] = (volume.iloc[i-3:i+1] > median_vol).sum()
    
    adaptive_reversal = reversal_1d * volatility_context * volume_persistence_3d
    
    # Volatility-Scaled Efficiency Momentum
    daily_efficiency = np.abs(close - close.shift(1)) / (high - low).replace(0, np.nan)
    
    three_day_efficiency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 3:
            price_change = np.abs(close.iloc[i] - close.iloc[i-3])
            high_low_sum = (high.iloc[i-3:i+1] - low.iloc[i-3:i+1]).sum()
            three_day_efficiency.iloc[i] = price_change / high_low_sum if high_low_sum > 0 else np.nan
    
    efficiency_ratio = daily_efficiency / three_day_efficiency
    
    short_term_vol = close.rolling(window=5).std()
    long_term_vol = close.rolling(window=10).std()
    volatility_regime = short_term_vol / long_term_vol
    
    volatility_scaled_momentum = momentum_5d / volatility_regime.replace(0, np.nan)
    efficiency_weighted_trend = momentum_5d * efficiency_ratio
    regime_adaptive_efficiency = efficiency_ratio * volatility_regime
    
    # Amount Flow Persistence Patterns
    net_directional_flow = amount * np.sign(close - close.shift(1))
    flow_momentum = net_directional_flow.rolling(window=3).sum()
    
    flow_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            flow_signs = np.sign(net_directional_flow.iloc[i-5:i+1])
            momentum_sign = np.sign(flow_momentum.iloc[i])
            flow_consistency.iloc[i] = (flow_signs == momentum_sign).sum()
    
    flow_acceleration = net_directional_flow / net_directional_flow.shift(1).replace(0, np.nan)
    flow_volatility = net_directional_flow.rolling(window=5).std()
    flow_to_volume = net_directional_flow / volume.replace(0, np.nan)
    
    sustained_directional_flow = flow_momentum * flow_consistency
    flow_breakout_detection = flow_acceleration * flow_volatility
    smart_money_persistence = flow_to_volume * flow_consistency
    
    # Range-Volume Regime Interaction
    range_expansion = (high - low) / (high.shift(1) - low.shift(1)).replace(0, np.nan)
    range_volatility = (high - low).rolling(window=5).std()
    range_efficiency = np.abs(close - close.shift(1)) / (high - low).replace(0, np.nan)
    
    volume_regime = volume / volume.rolling(window=10, min_periods=1).median()
    volume_persistence_5d = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            volume_regime_val = volume.rolling(window=10, min_periods=1).median().iloc[i]
            volume_persistence_5d.iloc[i] = (volume.iloc[i-5:i+1] > volume_regime_val).sum()
    
    volume_range_correlation = volume_regime * range_expansion
    
    # Combine all signals with appropriate weights
    factor = (
        0.15 * multi_timeframe_divergence +
        0.15 * adaptive_reversal +
        0.15 * volatility_scaled_momentum +
        0.15 * efficiency_weighted_trend +
        0.10 * sustained_directional_flow +
        0.10 * smart_money_persistence +
        0.10 * (volume_range_correlation * momentum_5d) +
        0.05 * flow_breakout_detection +
        0.05 * regime_adaptive_efficiency
    )
    
    return factor.fillna(0)
