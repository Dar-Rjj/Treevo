import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factors using volume-weighted momentum, intraday efficiency,
    multi-timeframe alignment, and composite volume integration approaches.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    alpha_factors = pd.Series(index=data.index, dtype=float)
    
    # Volume-Weighted Momentum Factors
    # Volume-Scaled Acceleration
    short_momentum = data['close'] / data['close'].shift(3) - 1
    medium_momentum = data['close'] / data['close'].shift(10) - 1
    volume_scaled_acceleration = (short_momentum - medium_momentum) * data['volume']
    
    # Volume-Confirmed Trend Strength
    price_trend = data['close'] / data['close'].shift(5) - 1
    volume_trend = data['volume'] / data['volume'].shift(5) - 1
    volume_weighted_confirmation = price_trend * volume_trend * data['volume']
    
    # Volume-Weighted Momentum Persistence
    return_days = pd.Series(index=data.index, dtype=float)
    volume_days = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        return_days.iloc[i] = sum(data['close'].iloc[i-j] > data['close'].iloc[i-j-1] for j in range(5))
        volume_days.iloc[i] = sum(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] for j in range(5))
    
    volume_weighted_persistence = return_days * volume_days * data['volume']
    
    # Intraday Efficiency Factors
    # Volume-Weighted Range Efficiency
    intraday_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    volume_weighted_efficiency = intraday_efficiency * data['volume']
    
    # Volume-Scaled Gap Momentum
    overnight_gap = data['open'] / data['close'].shift(1) - 1
    intraday_momentum = data['close'] / data['open'] - 1
    volume_scaled_gap_momentum = overnight_gap * intraday_momentum * data['volume']
    
    # Volume-Adjusted Price Impact
    price_change_magnitude = abs(data['close'] - data['close'].shift(1))
    volume_efficiency = data['volume'] / price_change_magnitude.replace(0, np.nan)
    volume_efficiency = volume_efficiency.replace([np.inf, -np.inf], np.nan)
    volume_weighted_impact = (1 / volume_efficiency) * data['volume']
    
    # Multi-Timeframe Alignment Factors
    # Volume-Weighted Trend Convergence
    short_trend = np.sign(data['close'] - data['close'].shift(3))
    medium_trend = np.sign(data['close'] - data['close'].shift(10))
    volume_weighted_alignment = short_trend * medium_trend * data['volume']
    
    # Volume-Scaled Range Expansion
    range_ratio = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    range_ratio = range_ratio.replace([np.inf, -np.inf], np.nan)
    volume_scaled_range = range_ratio * data['volume']
    
    # Volume-Weighted Efficiency Persistence
    efficiency_today = (data['close'] - data['open']) / (data['high'] - data['low'])
    efficiency_yesterday = (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    efficiency_change = efficiency_today - efficiency_yesterday
    volume_trend_daily = data['volume'] / data['volume'].shift(1) - 1
    volume_weighted_efficiency_persistence = efficiency_change * volume_trend_daily * data['volume']
    
    # Composite Volume Integration Factors
    # Volume-Momentum-Efficiency Composite
    momentum_component = data['close'] / data['close'].shift(5) - 1
    efficiency_component = (data['close'] - data['open']) / (data['high'] - data['low'])
    volume_intensity = data['volume'] / (data['high'] - data['low'])
    volume_intensity = volume_intensity.replace([np.inf, -np.inf], np.nan)
    composite_factor = momentum_component * efficiency_component * volume_intensity * data['volume']
    
    # Multi-Timeframe Volume Confirmation
    short_volume_momentum = data['volume'] / data['volume'].shift(1) - 1
    medium_volume_trend = data['volume'] / data['volume'].shift(5) - 1
    price_momentum = data['close'] / data['close'].shift(5) - 1
    multi_timeframe_confirmation = short_volume_momentum * medium_volume_trend * price_momentum * data['volume']
    
    # Volume-Weighted Mean Reversion
    price_position = (data['close'] - data['low']) / (data['high'] - data['low']) - 0.5
    price_position = price_position.replace([np.inf, -np.inf], np.nan)
    recent_momentum = data['close'] / data['close'].shift(3) - 1
    volume_weighted_reversion = price_position * volume_intensity * (-recent_momentum) * data['volume']
    
    # Combine all factors using equal weighting (can be optimized)
    factors = [
        volume_scaled_acceleration,
        volume_weighted_confirmation,
        volume_weighted_persistence,
        volume_weighted_efficiency,
        volume_scaled_gap_momentum,
        volume_weighted_impact,
        volume_weighted_alignment,
        volume_scaled_range,
        volume_weighted_efficiency_persistence,
        composite_factor,
        multi_timeframe_confirmation,
        volume_weighted_reversion
    ]
    
    # Normalize and combine factors
    normalized_factors = []
    for factor in factors:
        normalized = (factor - factor.mean()) / factor.std()
        normalized_factors.append(normalized)
    
    # Equal-weighted combination
    alpha_factors = sum(normalized_factors) / len(normalized_factors)
    
    return alpha_factors
