import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic price differences
    data['close_diff_1'] = data['close'].diff(1)
    data['close_diff_2'] = data['close'].diff(2)
    data['volume_diff_1'] = data['volume'].diff(1)
    data['volume_diff_2'] = data['volume'].diff(2)
    
    # Fracture Detection - Price Discontinuity
    data['gap_fracture'] = np.abs(data['open'] - data['close'].shift(1)) / (np.abs(data['close'].shift(1) - data['close'].shift(2)) + 1e-8)
    data['momentum_disruption'] = np.abs(data['close_diff_1'] - data['close_diff_1'].shift(1)) / (np.abs(data['close_diff_1'].shift(1)) + 1e-8)
    data['efficiency_fracture'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) - (np.abs(data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8))
    
    # Fracture Detection - Volume Fracture
    data['volume_shock'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) - (data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8))
    data['trade_size_break'] = np.abs((data['amount'] / (data['volume'] + 1e-8)) - (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8))) / (data['amount'].shift(1) / (data['volume'].shift(1) + 1e-8) + 1e-8)
    data['volume_price_fracture'] = np.abs((data['volume'] / (data['volume'].shift(1) + 1e-8)) - (data['close_diff_1'] / (data['close_diff_1'].shift(1) + 1e-8)))
    
    # Liquidity Asymmetry - Order Flow Imbalance
    data['net_fracture_pressure'] = (data['high'] - data['close']) - (data['close'] - data['low'])
    data['upside_fracture_volume'] = data['volume'] * (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['downside_fracture_volume'] = data['volume'] * (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Liquidity Asymmetry - Session Effects
    data['opening_liquidity'] = (data['open'] - data['low']) - (data['high'] - data['open'])
    data['closing_liquidity'] = (data['close'] - data['low']) - (data['high'] - data['close'])
    data['liquidity_momentum'] = data['net_fracture_pressure'] / (data['net_fracture_pressure'].shift(1) + 1e-8)
    
    # Fracture Transitions - Breakout Signals
    data['fracture_break_momentum'] = data['close_diff_1'] / (np.abs(data['close_diff_1'].shift(1)) + 1e-8)
    data['volatility_transition'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) * (data['close_diff_1'] / (np.abs(data['close_diff_1'].shift(1)) + 1e-8))
    data['volume_transition'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * (data['close_diff_1'] / (np.abs(data['close_diff_1'].shift(1)) + 1e-8))
    
    # Fracture Transitions - Acceleration Patterns
    data['momentum_acceleration'] = np.abs((data['close'] / (data['close'].shift(1) + 1e-8) - 1) - (data['close'].shift(1) / (data['close'].shift(2) + 1e-8) - 1)) / (np.abs(data['close'].shift(1) / (data['close'].shift(2) + 1e-8) - 1) + 1e-8)
    data['impact_fracture'] = (data['amount'] / (data['volume'] + 1e-8)) * (data['high'] - data['low']) / (np.abs(data['close'] - data['open']) + 1e-8)
    data['fracture_velocity'] = data['gap_fracture'] * data['fracture_break_momentum']
    
    # Fracture Validation - Multi-scale Alignment
    data['price_volume_alignment'] = np.sign(data['volume_diff_1']) * np.sign(data['close_diff_1'])
    data['volatility_price_alignment'] = np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))) * np.sign(data['close_diff_1'])
    data['efficiency_alignment'] = np.sign((np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) - (np.abs(data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8))) * np.sign(data['close_diff_1'])
    
    # Fracture Validation - Pattern Persistence
    for i in range(len(data)):
        if i >= 4:
            # Fracture Persistence
            fracture_count = 0
            for j in range(i-4, i+1):
                if j >= 3:
                    sign_current = np.sign(data['close'].iloc[j] - data['close'].iloc[j-1])
                    sign_previous = np.sign(data['close'].iloc[j-2] - data['close'].iloc[j-3])
                    if sign_current != sign_previous:
                        fracture_count += 1
            data.loc[data.index[i], 'fracture_persistence'] = fracture_count / 5.0
            
            # Liquidity Persistence
            liquidity_count = 0
            for j in range(i-4, i+1):
                if j >= 1:
                    if np.sign(data['net_fracture_pressure'].iloc[j]) == np.sign(data['net_fracture_pressure'].iloc[j-1]):
                        liquidity_count += 1
            data.loc[data.index[i], 'liquidity_persistence'] = liquidity_count / 5.0
            
            # Volume Persistence
            volume_count = 0
            for j in range(i-4, i+1):
                if j >= 1:
                    if np.abs(data['volume'].iloc[j] / (data['volume'].iloc[j-1] + 1e-8) - 1) > 0.5:
                        volume_count += 1
            data.loc[data.index[i], 'volume_persistence'] = volume_count / 5.0
        else:
            data.loc[data.index[i], 'fracture_persistence'] = 0
            data.loc[data.index[i], 'liquidity_persistence'] = 0
            data.loc[data.index[i], 'volume_persistence'] = 0
    
    # Alpha Synthesis - Core Factors
    data['liquidity_asymmetry'] = data['net_fracture_pressure'] * (data['upside_fracture_volume'] - data['downside_fracture_volume'])
    data['fracture_transition'] = data['volatility_transition'] * data['volume_transition']
    data['market_impact'] = data['impact_fracture'] * data['trade_size_break']
    data['fractal_break'] = data['momentum_acceleration'] * data['volume_price_fracture']
    
    # Alpha Synthesis - Validation Weights
    data['alignment_weight'] = (data['price_volume_alignment'] + data['volatility_price_alignment'] + data['efficiency_alignment']) / 3.0
    data['persistence_weight'] = (data['fracture_persistence'] + data['liquidity_persistence'] + data['volume_persistence']) / 3.0
    data['fracture_weight'] = 1 / (1 + data['gap_fracture'])
    
    # Alpha Synthesis - Final Alpha
    data['primary_alpha'] = data['liquidity_asymmetry'] * data['alignment_weight']
    data['secondary_alpha'] = data['fracture_transition'] * data['persistence_weight']
    data['tertiary_alpha'] = data['market_impact'] * data['fracture_weight']
    
    # Composite Alpha - Weighted combination
    data['composite_alpha'] = (0.4 * data['primary_alpha'] + 
                              0.35 * data['secondary_alpha'] + 
                              0.25 * data['tertiary_alpha'])
    
    # Return the composite alpha factor
    return data['composite_alpha']
