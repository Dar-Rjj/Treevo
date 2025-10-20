import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Breakout with Volume Asymmetry and Range Efficiency
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Short-Term Momentum (3-day)
    data['momentum_3d'] = (data['close'] / data['close'].shift(3) - 1) * 100
    
    # Medium-Term Momentum (10-day)
    data['momentum_10d'] = (data['close'] / data['close'].shift(10) - 1) * 100
    
    # Momentum Divergence Detection
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_10d']
    data['momentum_acceleration'] = data['momentum_divergence'] - data['momentum_divergence'].shift(1)
    
    # Volume Asymmetry Analysis
    # Directional Volume Response
    up_momentum_mask = data['momentum_3d'] > 0
    down_momentum_mask = data['momentum_3d'] < 0
    
    # Volume sensitivity calculations
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['volume_intensity'] = data['volume'] / data['volume_5d_avg']
    
    # Up-momentum volume response
    up_volume_response = data[up_momentum_mask]['volume_intensity'].rolling(window=5).mean()
    # Down-momentum volume response  
    down_volume_response = data[down_momentum_mask]['volume_intensity'].rolling(window=5).mean()
    
    # Asymmetry ratio (handle division by zero)
    data['volume_asymmetry'] = np.where(
        down_volume_response != 0,
        up_volume_response / down_volume_response,
        up_volume_response
    )
    data['volume_asymmetry'] = data['volume_asymmetry'].fillna(1.0)
    
    # Volume confirmation signals
    data['volume_confirmation'] = np.where(
        data['volume_intensity'] > 1.2,
        np.where(data['momentum_divergence'] > 0, 1, -1),
        0
    )
    
    # Range Efficiency Analysis
    # Daily Range Calculation
    data['daily_range'] = (data['high'] - data['low']) / data['close'] * 100
    data['range_5d_avg'] = data['daily_range'].rolling(window=5).mean()
    
    # Volatility-Adjusted Efficiency
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252) * 100
    data['range_efficiency'] = np.where(
        data['volatility_20d'] != 0,
        data['daily_range'] / data['volatility_20d'],
        0
    )
    
    # Breakout Strength Assessment
    data['range_expansion'] = data['daily_range'] / data['range_5d_avg'] - 1
    data['breakout_strength'] = data['range_expansion'] * data['range_efficiency']
    
    # Intraday Pressure Accumulation
    # Normalized Intraday Pressure
    data['intraday_pressure'] = np.where(
        data['high'] != data['low'],
        (data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    
    # Accumulate Pressure with Volume Weighting
    data['pressure_3d_cumulative'] = data['intraday_pressure'].rolling(window=3).sum()
    data['weighted_pressure'] = data['pressure_3d_cumulative'] * data['volume_asymmetry']
    
    # Pressure divergence detection
    data['pressure_divergence'] = data['weighted_pressure'] - data['momentum_3d'].rolling(window=3).mean()
    
    # Composite Alpha Generation
    # Multi-Dimensional Signal Integration
    momentum_component = data['momentum_divergence'] * data['volume_asymmetry']
    breakout_component = data['breakout_strength'] * data['range_efficiency']
    pressure_component = data['weighted_pressure'] * np.sign(data['momentum_acceleration'])
    
    # Signal Validation and Coherence
    volume_alignment = np.where(
        np.sign(data['volume_confirmation']) == np.sign(data['momentum_divergence']),
        np.abs(data['volume_confirmation']),
        0
    )
    
    range_alignment = np.where(
        np.sign(data['breakout_strength']) == np.sign(data['momentum_divergence']),
        np.abs(data['breakout_strength']),
        0
    )
    
    pressure_alignment = np.where(
        np.sign(data['pressure_divergence']) == np.sign(data['momentum_divergence']),
        np.abs(data['pressure_divergence']),
        0
    )
    
    # Combined signal strength
    signal_strength = (volume_alignment + range_alignment + pressure_alignment) / 3
    
    # Final Predictive Factor
    alpha = (
        momentum_component * 0.4 +
        breakout_component * 0.3 +
        pressure_component * 0.3
    ) * signal_strength
    
    # Apply volatility normalization
    volatility_normalized = data['volatility_20d'].rolling(window=20).mean()
    final_alpha = np.where(
        volatility_normalized != 0,
        alpha / volatility_normalized,
        alpha
    )
    
    return final_alpha
