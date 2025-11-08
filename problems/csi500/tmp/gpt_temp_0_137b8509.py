import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Calculation
    # Short-Term Momentum (1-day)
    data['short_momentum_raw'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['short_momentum_direction'] = np.sign(data['close'] - data['open'])
    data['short_momentum_strength'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    
    # Medium-Term Momentum (3-day)
    data['medium_price_change'] = data['close'] / data['close'].shift(3) - 1
    data['medium_volatility'] = (data['high'] - data['low']).rolling(window=3, min_periods=1).mean()
    data['medium_momentum'] = data['medium_price_change'] / (data['medium_volatility'] + epsilon)
    
    # Long-Term Momentum (5-day)
    data['long_price_change'] = data['close'] / data['close'].shift(5) - 1
    data['long_volatility'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['long_momentum'] = data['long_price_change'] / (data['long_volatility'] + epsilon)
    
    # Volume Persistence Analysis
    # Volume Trend Detection
    data['volume_ratio_3d'] = data['volume'] / (data['volume'].shift(3) + epsilon)
    data['volume_ratio_5d'] = data['volume'] / (data['volume'].shift(5) + epsilon)
    data['volume_acceleration'] = ((data['volume'] / (data['volume'].shift(1) + epsilon)) / 
                                 (data['volume'].shift(1) / (data['volume'].shift(2) + epsilon) + epsilon))
    
    # Volume-Momentum Alignment
    data['short_volume_alignment'] = (np.sign(data['short_momentum_raw']) * 
                                    np.sign(data['volume_ratio_3d']))
    data['medium_volume_alignment'] = (np.sign(data['medium_momentum']) * 
                                     np.sign(data['volume_ratio_3d']))
    data['alignment_strength'] = np.minimum(
        np.abs(data['short_momentum_raw']), 
        np.abs(data['volume_ratio_3d'])
    )
    
    # Signal Combination with Exponential Decay
    decay_factors = {'short': 0.9, 'medium': 0.95, 'long': 0.98}
    
    # Apply exponential decay to momentum signals
    data['short_momentum_decayed'] = data['short_momentum_raw']
    data['medium_momentum_decayed'] = data['medium_momentum']
    data['long_momentum_decayed'] = data['long_momentum']
    
    # Simple exponential smoothing (can be enhanced with proper EWMA)
    for i in range(1, len(data)):
        if not pd.isna(data.loc[data.index[i-1], 'short_momentum_decayed']):
            data.loc[data.index[i], 'short_momentum_decayed'] = (
                decay_factors['short'] * data.loc[data.index[i-1], 'short_momentum_decayed'] + 
                (1 - decay_factors['short']) * data.loc[data.index[i], 'short_momentum_raw']
            )
        if not pd.isna(data.loc[data.index[i-1], 'medium_momentum_decayed']):
            data.loc[data.index[i], 'medium_momentum_decayed'] = (
                decay_factors['medium'] * data.loc[data.index[i-1], 'medium_momentum_decayed'] + 
                (1 - decay_factors['medium']) * data.loc[data.index[i], 'medium_momentum']
            )
        if not pd.isna(data.loc[data.index[i-1], 'long_momentum_decayed']):
            data.loc[data.index[i], 'long_momentum_decayed'] = (
                decay_factors['long'] * data.loc[data.index[i-1], 'long_momentum_decayed'] + 
                (1 - decay_factors['long']) * data.loc[data.index[i], 'long_momentum']
            )
    
    # Combine timeframes with weights
    weights = {'short': 0.4, 'medium': 0.35, 'long': 0.25}
    data['combined_momentum'] = (
        weights['short'] * data['short_momentum_decayed'] +
        weights['medium'] * data['medium_momentum_decayed'] +
        weights['long'] * data['long_momentum_decayed']
    )
    
    # Apply volume alignment
    data['volume_adjusted_signal'] = (
        data['combined_momentum'] * 
        data['alignment_strength'] * 
        np.sign(data['short_volume_alignment'])
    )
    
    # Volatility Scaling and Final Output
    # Dynamic Volatility Adjustment
    data['recent_volatility'] = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    data['historical_volatility'] = (data['high'] - data['low']).rolling(window=20, min_periods=1).mean()
    data['volatility_ratio'] = data['recent_volatility'] / (data['historical_volatility'] + epsilon)
    
    # Final Factor Construction
    # Scale by inverse volatility ratio and apply dampening
    volatility_dampening = np.exp(-np.abs(data['volatility_ratio'] - 1))
    data['final_factor'] = (
        data['volume_adjusted_signal'] / 
        (data['volatility_ratio'] + epsilon) * 
        volatility_dampening
    )
    
    return data['final_factor']
