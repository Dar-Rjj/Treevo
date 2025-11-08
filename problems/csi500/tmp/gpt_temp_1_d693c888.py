import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining volume-weighted price efficiency, 
    multi-timeframe trend confirmation, volume-enhanced price impact, and 
    multi-scale volume integration.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volume-Weighted Price Efficiency
    # Volume-Adjusted Intraday Efficiency
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    volume_adjusted_efficiency = intraday_efficiency * df['volume']
    
    # Volume-Scaled Gap Persistence
    overnight_gap = df['open'] / df['close'].shift(1) - 1
    gap_persistence = overnight_gap * overnight_gap.shift(1)
    volume_scaled_gap = gap_persistence * df['volume']
    
    # Volume-Weighted Range Utilization
    range_utilization = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    range_utilization = range_utilization.replace([np.inf, -np.inf], np.nan)
    volume_efficiency = df['volume'] / (df['high'] - df['low'])
    volume_efficiency = volume_efficiency.replace([np.inf, -np.inf], np.nan)
    combined_range_factor = range_utilization * volume_efficiency * df['volume']
    
    # Multi-Timeframe Trend Confirmation
    # Volume-Weighted Trend Alignment
    short_trend = df['close'] / df['close'].shift(3) - 1
    medium_trend = df['close'] / df['close'].shift(10) - 1
    trend_alignment = short_trend * medium_trend * df['volume']
    
    # Volume-Confirmed Momentum Acceleration
    momentum_acceleration = (df['close']/df['close'].shift(3) - 1) - (df['close']/df['close'].shift(10) - 1)
    volume_momentum = df['volume'] / df['volume'].shift(3) - 1
    volume_confirmed_acceleration = momentum_acceleration * volume_momentum * df['volume']
    
    # Volume-Weighted Trend Persistence
    price_persistence = pd.Series(index=df.index, dtype=float)
    volume_persistence = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        price_window = df['close'].iloc[i-5:i]
        volume_window = df['volume'].iloc[i-5:i]
        price_persistence.iloc[i] = (price_window > price_window.shift(1)).sum()
        volume_persistence.iloc[i] = (volume_window > volume_window.shift(1)).sum()
    
    volume_weighted_persistence = price_persistence * volume_persistence * df['volume']
    
    # Volume-Enhanced Price Impact
    # Volume-Weighted Price Momentum
    price_momentum = df['close'] / df['close'].shift(1) - 1
    volume_intensity = df['volume'] / abs(df['close'] - df['close'].shift(1))
    volume_intensity = volume_intensity.replace([np.inf, -np.inf], np.nan)
    volume_weighted_momentum = price_momentum * volume_intensity * df['volume']
    
    # Volume-Scaled Range Expansion
    range_expansion = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan)
    volume_expansion = df['volume'] / df['volume'].shift(1)
    volume_scaled_expansion = range_expansion * volume_expansion * df['volume']
    
    # Volume-Adjusted Gap Efficiency
    gap_efficiency = (df['close'] - df['open']) / abs(df['open'] - df['close'].shift(1))
    gap_efficiency = gap_efficiency.replace([np.inf, -np.inf], np.nan)
    volume_per_gap = df['volume'] / abs(df['open'] - df['close'].shift(1))
    volume_per_gap = volume_per_gap.replace([np.inf, -np.inf], np.nan)
    volume_adjusted_efficiency = gap_efficiency * volume_per_gap * df['volume']
    
    # Multi-Scale Volume Integration
    # Short-Term Volume-Price Synergy
    price_change = df['close'] / df['close'].shift(1) - 1
    volume_change = df['volume'] / df['volume'].shift(1) - 1
    volume_price_synergy = price_change * volume_change * df['volume']
    
    # Medium-Term Volume Consistency
    price_consistency = df['close'] / df['close'].shift(5) - 1
    volume_consistency = df['volume'] / df['volume'].shift(5) - 1
    volume_weighted_consistency = price_consistency * volume_consistency * df['volume']
    
    # Volume-Weighted Mean Reversion
    intraday_position = (df['close'] - df['low']) / (df['high'] - df['low']) - 0.5
    intraday_position = intraday_position.replace([np.inf, -np.inf], np.nan)
    volume_intensity_reversion = df['volume'] / (df['high'] - df['low'])
    volume_intensity_reversion = volume_intensity_reversion.replace([np.inf, -np.inf], np.nan)
    volume_weighted_reversion = intraday_position * volume_intensity_reversion * df['volume']
    
    # Combine all factors with equal weights
    factors = [
        volume_adjusted_efficiency,
        volume_scaled_gap,
        combined_range_factor,
        trend_alignment,
        volume_confirmed_acceleration,
        volume_weighted_persistence,
        volume_weighted_momentum,
        volume_scaled_expansion,
        volume_adjusted_efficiency,
        volume_price_synergy,
        volume_weighted_consistency,
        volume_weighted_reversion
    ]
    
    # Normalize and combine factors
    normalized_factors = []
    for factor in factors:
        normalized_factor = (factor - factor.mean()) / factor.std()
        normalized_factors.append(normalized_factor)
    
    # Equal-weighted combination
    result = sum(normalized_factors) / len(normalized_factors)
    
    return result
