import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence System
    # Intraday Momentum Divergence
    price_momentum = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    volume_momentum = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1).replace(0, np.nan)
    intraday_divergence = price_momentum * volume_momentum
    
    # Gap Asymmetry Analysis
    morning_gap = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    evening_gap = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    gap_ratio = morning_gap / evening_gap.replace(0, np.nan)
    
    price_volume_divergence = intraday_divergence * gap_ratio
    
    # Volatility Compression Dynamics
    # Range Compression
    current_range = data['high'] - data['low']
    historical_high = data['high'].rolling(window=10, min_periods=1).max()
    historical_low = data['low'].rolling(window=10, min_periods=1).min()
    historical_range = historical_high - historical_low
    compression_ratio = current_range / historical_range.replace(0, np.nan)
    
    # Volume Compression
    current_volume = data['volume']
    avg_volume = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_ratio = current_volume / avg_volume.replace(0, np.nan)
    
    volatility_compression = compression_ratio * volume_ratio
    
    # Behavioral Pattern Recognition
    # Price Pattern Strength
    price_change = data['close'].diff()
    consecutive_direction = price_change.rolling(window=5).apply(
        lambda x: sum((x > 0) == (x.iloc[-1] > 0)) if len(x) == 5 else np.nan, raw=False
    )
    
    price_magnitude = price_change.abs().rolling(window=5, min_periods=1).sum()
    price_range_5d = data['close'].rolling(window=5, min_periods=1).max() - data['close'].rolling(window=5, min_periods=1).min()
    pattern_magnitude = price_magnitude / price_range_5d.replace(0, np.nan)
    pattern_score = consecutive_direction * pattern_magnitude
    
    # Volume Pattern Recognition
    volume_spike = data['volume'] / data['volume'].rolling(window=9, min_periods=1).max().shift(1).replace(0, np.nan)
    volume_consistency = data['volume'].rolling(window=5, min_periods=1).std() / data['volume'].rolling(window=5, min_periods=1).mean().replace(0, np.nan)
    volume_pattern = volume_spike * (1 - volume_consistency)
    
    behavioral_pattern = pattern_score * volume_pattern
    
    # Market Microstructure Analysis
    # Price Efficiency Measure
    intraday_efficiency = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    overnight_efficiency = (data['open'] - data['close'].shift(1)).abs() / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    efficiency_ratio = intraday_efficiency / overnight_efficiency.replace(0, np.nan)
    
    # Volume-Price Alignment
    bullish_alignment = ((data['close'] > data['open']) & (data['volume'] > data['volume'].shift(1))).astype(float)
    bearish_alignment = ((data['close'] < data['open']) & (data['volume'] > data['volume'].shift(1))).astype(float)
    alignment_score = bullish_alignment - bearish_alignment
    
    microstructure_analysis = efficiency_ratio * alignment_score
    
    # Multi-Timeframe Integration
    # Short-Term Dynamics (1-2 days)
    price_change_st = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    volume_change_st = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1).replace(0, np.nan)
    short_term_factor = price_change_st * volume_change_st
    
    # Medium-Term Dynamics (3-5 days)
    price_momentum_mt = (data['close'] - data['close'].shift(3)) / (
        data['high'].rolling(window=4, min_periods=1).max() - data['low'].rolling(window=4, min_periods=1).min()
    ).replace(0, np.nan)
    volume_trend_mt = data['volume'] / data['volume'].shift(3).replace(0, np.nan)
    medium_term_factor = price_momentum_mt * volume_trend_mt
    
    multi_timeframe_integration = short_term_factor * medium_term_factor
    
    # Dynamic Weighting
    volatility_condition = (data['high'] - data['low']) / (
        data['high'].rolling(window=5, min_periods=1).max() - data['low'].rolling(window=5, min_periods=1).min()
    ).replace(0, np.nan)
    
    trend_condition = (data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=6, min_periods=1).max() - data['low'].rolling(window=6, min_periods=1).min()
    ).replace(0, np.nan)
    
    weight_adjustment = volatility_condition * trend_condition
    
    # Final Alpha Construction with Dynamic Component Selection
    # Component selection based on market conditions
    selected_component = (
        price_volume_divergence * (volatility_condition > 1) +  # High volatility
        volatility_compression * (volatility_condition <= 1) * (trend_condition.abs() <= 0.02) +  # Low volatility, no trend
        behavioral_pattern * (trend_condition.abs() > 0.02) +  # Trending market
        microstructure_analysis * (volatility_condition <= 1) * (trend_condition.abs() <= 0.02)  # Range-bound market
    )
    
    final_alpha = selected_component * multi_timeframe_integration * weight_adjustment
    
    return final_alpha
