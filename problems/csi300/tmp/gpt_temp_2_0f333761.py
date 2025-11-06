import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily ranges
    data['range'] = data['high'] - data['low']
    data['prev_range'] = data['range'].shift(1)
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    
    # Volatility State Identification
    # Expansion Detection
    expansion_count = pd.Series(index=data.index, dtype=float)
    contraction_count = pd.Series(index=data.index, dtype=float)
    
    for i in range(4, len(data)):
        window_data = data.iloc[i-4:i+1]
        expansion_mask = window_data['range'] > 1.5 * window_data['prev_range']
        contraction_mask = window_data['range'] < window_data['prev_range']
        
        expansion_count.iloc[i] = expansion_mask.sum()
        contraction_count.iloc[i] = contraction_mask.sum()
    
    data['expansion_score'] = expansion_count / 4
    data['contraction_score'] = contraction_count / 4
    data['regime_signal'] = data['expansion_score'] - data['contraction_score']
    
    # Price-Volume Divergence Analysis
    data['price_change'] = data['close'] - data['prev_close']
    data['volume_change'] = data['volume'] - data['prev_volume']
    
    data['price_sign'] = np.sign(data['price_change'])
    data['volume_sign'] = np.sign(data['volume_change'])
    
    # Calculate divergences
    negative_div = pd.Series(index=data.index, dtype=float)
    positive_div = pd.Series(index=data.index, dtype=float)
    
    mismatch_mask = data['price_sign'] != data['volume_sign']
    match_mask = data['price_sign'] == data['volume_sign']
    
    negative_div[mismatch_mask] = data['price_change'][mismatch_mask] / data['volume'][mismatch_mask]
    positive_div[match_mask] = data['price_change'][match_mask] / data['volume'][match_mask]
    
    # Fill NaN values with 0
    negative_div = negative_div.fillna(0)
    positive_div = positive_div.fillna(0)
    
    data['divergence_intensity'] = np.abs(negative_div - positive_div)
    
    # Multi-Timeframe Integration
    # Short-term Component
    data['short_term_component'] = (
        data['regime_signal'] * 
        data['price_change'] * 
        data['range'] / data['prev_range']
    ).fillna(0)
    
    # Medium-term Component
    regime_sum = pd.Series(index=data.index, dtype=float)
    volume_weighted_regime = pd.Series(index=data.index, dtype=float)
    volume_sum = pd.Series(index=data.index, dtype=float)
    
    for i in range(4, len(data)):
        window_data = data.iloc[i-4:i+1]
        regime_sum.iloc[i] = window_data['regime_signal'].sum()
        volume_weighted_regime.iloc[i] = (window_data['volume'] * window_data['regime_signal']).sum()
        volume_sum.iloc[i] = window_data['volume'].sum()
    
    data['medium_term_component'] = (regime_sum / 5) * (volume_weighted_regime / volume_sum.replace(0, 1))
    data['medium_term_component'] = data['medium_term_component'].fillna(0)
    
    # Adaptive Signal Enhancement
    data['expansion_multiplier'] = 1 + data['expansion_score']
    data['contraction_multiplier'] = 1 + data['contraction_score']
    data['transition_multiplier'] = 1 + np.abs(data['regime_signal'])
    
    # Alpha Construction
    data['core_signal'] = (
        data['divergence_intensity'] * 
        (data['expansion_multiplier'] + data['contraction_multiplier'] + data['transition_multiplier'])
    )
    
    data['multi_scale_signal'] = (
        data['core_signal'] * 
        data['short_term_component'] * 
        data['medium_term_component']
    )
    
    data['final_alpha'] = data['multi_scale_signal'] / (1 + np.abs(data['regime_signal']))
    
    # Return the final alpha factor
    return data['final_alpha']
