import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Momentum Decay-Adjusted Trend Strength
    # Rolling Momentum with Exponential Decay
    momentum_periods = [5, 10, 20]
    decay_weights = np.exp(-np.arange(max(momentum_periods)) * 0.1)
    
    momentum_components = []
    for period in momentum_periods:
        weights = decay_weights[:period] / decay_weights[:period].sum()
        weighted_momentum = sum(weights[i] * (data['close'] - data['close'].shift(i+1)) 
                              for i in range(period))
        momentum_components.append(weighted_momentum)
    
    momentum_signal = pd.concat(momentum_components, axis=1).mean(axis=1)
    
    # Volatility Adjustment
    volatility_window = 20
    high_low_range = (data['high'] - data['low']) / data['close']
    rolling_volatility = high_low_range.rolling(window=volatility_window).std()
    
    momentum_strength = momentum_signal / (rolling_volatility + 1e-8)
    
    # Volume-Price Divergence Oscillator
    price_roc = data['close'].pct_change(periods=5)
    volume_roc = data['volume'].pct_change(periods=5)
    
    # Normalize both series
    price_roc_norm = (price_roc - price_roc.rolling(20).mean()) / (price_roc.rolling(20).std() + 1e-8)
    volume_roc_norm = (volume_roc - volume_roc.rolling(20).mean()) / (volume_roc.rolling(20).std() + 1e-8)
    
    divergence_signal = price_roc_norm - volume_roc_norm
    
    # Bid-Ask Spread Implied Pressure
    spread_proxy = (data['high'] - data['low']) / data['close']
    spread_change = spread_proxy.diff()
    volume_change = data['volume'].pct_change()
    
    # Spread-Volume Interaction
    consensus_signal = ((volume_change > 0) & (spread_change < 0)).astype(int)
    disagreement_signal = ((volume_change > 0) & (spread_change > 0)).astype(int)
    
    spread_pressure = consensus_signal.rolling(5).sum() - disagreement_signal.rolling(5).sum()
    
    # Opening Gap Mean Reversion
    gap_size = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_magnitude = gap_size.abs()
    
    # Volume-weighted gap persistence
    volume_quantile = data['volume'].rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    gap_persistence = gap_magnitude * (1 - volume_quantile)
    mean_reversion_prob = -gap_persistence  # Negative for mean reversion
    
    # Liquidity-Confirmed Breakout
    ma_fast = data['close'].rolling(window=10).mean()
    ma_slow = data['close'].rolling(window=20).mean()
    
    # Breakout signals
    price_above_fast = data['close'] > ma_fast
    price_above_slow = data['close'] > ma_slow
    new_high = data['close'] == data['close'].rolling(20).max()
    
    # Volume confirmation
    volume_ma = data['volume'].rolling(20).mean()
    high_volume = data['volume'] > volume_ma
    
    validated_breakout = ((price_above_fast & price_above_slow & new_high) & high_volume).astype(int)
    
    # Range Cycle Phase Detection
    daily_range = (data['high'] - data['low']) / data['close']
    range_ma = daily_range.rolling(20).mean()
    range_std = daily_range.rolling(20).std()
    
    range_zscore = (daily_range - range_ma) / (range_std + 1e-8)
    
    # Cycle timing signals
    contraction_signal = (range_zscore < -1).astype(int)  # Extended contraction
    expansion_signal = (range_zscore > 1).astype(int)     # Rapid expansion
    
    cycle_timing = contraction_signal - expansion_signal
    
    # Volume-Weighted Price Levels
    # Identify significant price levels (support/resistance)
    lookback = 50
    support_levels = data['low'].rolling(lookback).min()
    resistance_levels = data['high'].rolling(lookback).max()
    
    # Distance to key levels
    dist_to_support = (data['close'] - support_levels) / data['close']
    dist_to_resistance = (resistance_levels - data['close']) / data['close']
    
    # Volume-weighted level significance
    recent_volume_weight = data['volume'].rolling(10).mean() / data['volume'].rolling(50).mean()
    level_strength = (dist_to_support * recent_volume_weight) - (dist_to_resistance * recent_volume_weight)
    
    # Combine all signals with appropriate weights
    factor = (
        0.25 * momentum_strength.rank(pct=True) +
        0.20 * divergence_signal.rank(pct=True) +
        0.15 * spread_pressure.rank(pct=True) +
        0.15 * mean_reversion_prob.rank(pct=True) +
        0.10 * validated_breakout.rank(pct=True) +
        0.10 * cycle_timing.rank(pct=True) +
        0.05 * level_strength.rank(pct=True)
    )
    
    return factor
