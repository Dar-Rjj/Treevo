import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Congruence Factor
    Combines multi-timeframe momentum with volume analysis and regime detection
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    # Price momentum calculations
    data['price_momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['price_momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['price_momentum_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    
    # Volume momentum calculations
    data['volume_momentum_5d'] = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    data['volume_momentum_10d'] = (data['volume'] - data['volume'].shift(10)) / (data['volume'].shift(10) + 1e-8)
    data['volume_momentum_20d'] = (data['volume'] - data['volume'].shift(20)) / (data['volume'].shift(20) + 1e-8)
    
    # Exponential Smoothing Application
    # Initialize smoothed momentum columns
    data['smooth_price_5d'] = data['price_momentum_5d']
    data['smooth_price_10d'] = data['price_momentum_10d']
    data['smooth_price_20d'] = data['price_momentum_20d']
    data['smooth_volume_5d'] = data['volume_momentum_5d']
    data['smooth_volume_10d'] = data['volume_momentum_10d']
    data['smooth_volume_20d'] = data['volume_momentum_20d']
    
    # Apply exponential smoothing
    alpha_short, alpha_medium, alpha_long = 0.3, 0.15, 0.1
    
    for i in range(1, len(data)):
        # Price momentum smoothing
        data.loc[data.index[i], 'smooth_price_5d'] = (alpha_short * data.loc[data.index[i], 'price_momentum_5d'] + 
                                                     (1 - alpha_short) * data.loc[data.index[i-1], 'smooth_price_5d'])
        data.loc[data.index[i], 'smooth_price_10d'] = (alpha_medium * data.loc[data.index[i], 'price_momentum_10d'] + 
                                                      (1 - alpha_medium) * data.loc[data.index[i-1], 'smooth_price_10d'])
        data.loc[data.index[i], 'smooth_price_20d'] = (alpha_long * data.loc[data.index[i], 'price_momentum_20d'] + 
                                                      (1 - alpha_long) * data.loc[data.index[i-1], 'smooth_price_20d'])
        
        # Volume momentum smoothing
        data.loc[data.index[i], 'smooth_volume_5d'] = (alpha_short * data.loc[data.index[i], 'volume_momentum_5d'] + 
                                                      (1 - alpha_short) * data.loc[data.index[i-1], 'smooth_volume_5d'])
        data.loc[data.index[i], 'smooth_volume_10d'] = (alpha_medium * data.loc[data.index[i], 'volume_momentum_10d'] + 
                                                       (1 - alpha_medium) * data.loc[data.index[i-1], 'smooth_volume_10d'])
        data.loc[data.index[i], 'smooth_volume_20d'] = (alpha_long * data.loc[data.index[i], 'volume_momentum_20d'] + 
                                                       (1 - alpha_long) * data.loc[data.index[i-1], 'smooth_volume_20d'])
    
    # Momentum Decay Assessment
    data['momentum_decay_short'] = np.abs(data['smooth_price_5d'] - data['smooth_price_10d'])
    data['momentum_decay_medium'] = np.abs(data['smooth_price_10d'] - data['smooth_price_20d'])
    data['momentum_decay_ratio'] = data['momentum_decay_short'] / (data['momentum_decay_medium'] + 1e-8)
    
    # Volume Acceleration Analysis
    data['volume_acceleration_5_10'] = data['smooth_volume_5d'] - data['smooth_volume_10d']
    data['volume_acceleration_10_20'] = data['smooth_volume_10d'] - data['smooth_volume_20d']
    data['volume_confirmation'] = np.sign(data['smooth_price_5d']) * data['volume_acceleration_5_10']
    
    # Regime Detection and Weighting
    # Calculate momentum consistency across timeframes
    data['momentum_direction_5d'] = np.sign(data['smooth_price_5d'])
    data['momentum_direction_10d'] = np.sign(data['smooth_price_10d'])
    data['momentum_direction_20d'] = np.sign(data['smooth_price_20d'])
    
    # Regime classification
    data['momentum_consistency'] = (data['momentum_direction_5d'] == data['momentum_direction_10d']).astype(int) + \
                                  (data['momentum_direction_10d'] == data['momentum_direction_20d']).astype(int)
    
    # Define regimes
    data['regime_trending'] = (data['momentum_consistency'] == 2).astype(float)
    data['regime_mean_reverting'] = (data['momentum_consistency'] == 0).astype(float)
    data['regime_transition'] = (data['momentum_consistency'] == 1).astype(float)
    
    # Final Factor Construction
    # Price-volume congruence score
    data['price_volume_congruence'] = (data['smooth_price_5d'] * np.sign(data['smooth_volume_5d']) + 
                                      data['smooth_price_10d'] * np.sign(data['smooth_volume_10d']) + 
                                      data['smooth_price_20d'] * np.sign(data['smooth_volume_20d'])) / 3
    
    # Momentum decay adjustment
    decay_adjustment = 1 - (data['momentum_decay_ratio'] / (1 + data['momentum_decay_ratio']))
    
    # Volume acceleration multiplier
    volume_multiplier = 1 + (data['volume_confirmation'] * 0.5)
    
    # Combine components with regime-specific weights
    trending_weight, mean_reverting_weight, transition_weight = 0.6, 0.3, 0.1
    
    base_factor = data['price_volume_congruence'] * decay_adjustment * volume_multiplier
    
    # Apply regime-specific final weighting
    final_factor = (data['regime_trending'] * trending_weight + 
                   data['regime_mean_reverting'] * mean_reverting_weight + 
                   data['regime_transition'] * transition_weight) * base_factor
    
    # Clean up and return
    result = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
