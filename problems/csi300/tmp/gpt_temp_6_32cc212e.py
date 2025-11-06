import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Regime Patterns
    data['momentum_asymmetry'] = ((data['close'] - data['close'].shift(1)) * 
                                 (data['high'] - data['low']) / 
                                 (data['high'].shift(1) - data['low'].shift(1)))
    
    data['multi_day_divergence'] = ((data['close'] / data['close'].shift(5) - 1) - 
                                   (data['close'].shift(2) / data['close'].shift(7) - 1))
    
    data['volatility_asymmetry'] = ((data['high'] - data['low']) / 
                                   (data['high'].shift(2) - data['low'].shift(2)) * 
                                   np.sign(data['close'] - data['close'].shift(1)))
    
    # Volume Regime Integration
    data['volume_coherence'] = ((data['volume'] - data['volume'].shift(2)) * 
                               np.sign(data['close'] - data['close'].shift(2)))
    
    data['multi_day_volume'] = ((data['volume'] / data['volume'].shift(4) - 1) * 
                               (data['close'] / data['close'].shift(4) - 1))
    
    data['volume_weighted_regime'] = (np.abs(data['close'] - data['close'].shift(2)) * 
                                     data['volume'] / 
                                     (data['high'].shift(1) - data['low'].shift(1)))
    
    # Price Transition Patterns
    data['intraday_transition'] = (np.abs(data['close'] - data['open']) / 
                                  (data['high'].shift(1) - data['low'].shift(1)))
    
    # Multi-day transition with rolling window
    data['rolling_high_3d'] = data['high'].rolling(window=4, min_periods=1).max()
    data['rolling_low_3d'] = data['low'].rolling(window=4, min_periods=1).min()
    data['multi_day_transition'] = (np.abs(data['close'] - data['close'].shift(4)) / 
                                   (data['rolling_high_3d'] - data['rolling_low_3d']))
    
    data['transition_reversal'] = (data['intraday_transition'] * 
                                  (1 - data['multi_day_transition']) * 
                                  np.sign(data['close'] - data['close'].shift(2)))
    
    # Volume Transition
    data['volume_transition'] = ((data['volume'] / data['volume'].shift(2)) * 
                                np.abs(data['close'] - data['close'].shift(2)) / 
                                (data['high'].shift(1) - data['low'].shift(1)))
    
    data['volume_price_transition'] = data['volume_transition'] * data['intraday_transition']
    
    # Regime-Flow Integration
    data['accumulation'] = (data['close'] - data['close'].shift(2)) * data['volume']
    data['multi_day_flow'] = ((data['close'] - data['close'].shift(4)) * 
                             (data['volume'] - data['volume'].shift(4)))
    
    data['gap_absorption'] = (np.sign(data['open'] - data['close'].shift(2)) * 
                             np.sign(data['close'] - data['open']))
    
    data['gap_flow'] = (data['close'] - data['open']) * data['volume'] * data['gap_absorption']
    
    # Multi-scale Validation - Regime Consistency
    def calculate_price_alignment(row_idx, data):
        if row_idx < 4:
            return np.nan
        count = 0
        for i in range(row_idx-4, row_idx+1):
            if i >= 2 and i-2 >= 0 and i-4 >= 0:
                sign1 = np.sign(data.iloc[i]['close'] - data.iloc[i-2]['close'])
                sign2 = np.sign(data.iloc[i-2]['close'] - data.iloc[i-4]['close'])
                if sign1 == sign2:
                    count += 1
        return count / 3
    
    def calculate_volume_stability(row_idx, data):
        if row_idx < 4:
            return np.nan
        count = 0
        for i in range(row_idx-4, row_idx+1):
            if i >= 2 and i-2 >= 0:
                vol_cond = data.iloc[i]['volume'] > data.iloc[i-2]['volume']
                price_cond = data.iloc[i]['close'] > data.iloc[i-2]['close']
                if vol_cond and price_cond:
                    count += 1
        return count / 3
    
    def calculate_flow_persistence(row_idx, data):
        if row_idx < 4:
            return np.nan
        count = 0
        for i in range(row_idx-4, row_idx+1):
            if i >= 2 and i-2 >= 0:
                if (data.iloc[i]['accumulation'] * data.iloc[i-2]['accumulation']) > 0:
                    count += 1
        return count / 3
    
    def calculate_regime_validation(row_idx, data):
        if row_idx < 4:
            return np.nan
        count = 0
        for i in range(row_idx-4, row_idx+1):
            if (data.iloc[i]['momentum_asymmetry'] * data.iloc[i]['volume_coherence']) > 0:
                count += 1
        return count / 3
    
    # Calculate validation metrics
    price_alignment_vals = []
    volume_stability_vals = []
    flow_persistence_vals = []
    regime_validation_vals = []
    
    for i in range(len(data)):
        price_alignment_vals.append(calculate_price_alignment(i, data))
        volume_stability_vals.append(calculate_volume_stability(i, data))
        flow_persistence_vals.append(calculate_flow_persistence(i, data))
        regime_validation_vals.append(calculate_regime_validation(i, data))
    
    data['price_alignment'] = price_alignment_vals
    data['volume_stability'] = volume_stability_vals
    data['flow_persistence'] = flow_persistence_vals
    data['regime_validation'] = regime_validation_vals
    
    # Alpha Synthesis
    data['price_regime'] = data['momentum_asymmetry'] * data['price_alignment']
    data['volume_regime'] = data['volume_coherence'] * data['volume_stability']
    data['flow_regime'] = data['accumulation'] * data['flow_persistence']
    data['gap_regime'] = data['gap_flow'] * data['gap_absorption']
    
    # Final Alpha Construction
    data['primary_factor'] = data['price_regime'] * data['volume_regime']
    data['secondary_factor'] = data['flow_regime'] * data['gap_regime']
    data['composite_alpha'] = data['primary_factor'] * data['secondary_factor'] * data['regime_validation']
    
    return data['composite_alpha']
