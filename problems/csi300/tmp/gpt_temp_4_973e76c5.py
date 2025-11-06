import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Fracture
    data['short_term_break'] = (data['close'] - data['close'].shift(2)) - (data['close'].shift(2) - data['close'].shift(4))
    data['medium_term_break'] = (data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))
    data['momentum_fracture_score'] = (
        np.sign(data['short_term_break']) * 
        np.sign(data['medium_term_break']) * 
        (np.abs(data['short_term_break']) - np.abs(data['medium_term_break']))
    )
    
    # Volume Momentum Fracture
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(1)) - (data['volume'].shift(1) - data['volume'].shift(2))
    data['volume_deceleration'] = (data['volume'].shift(1) - data['volume'].shift(2)) - (data['volume'].shift(2) - data['volume'].shift(3))
    data['volume_fracture_score'] = (
        np.sign(data['volume_acceleration']) * 
        np.sign(data['volume_deceleration']) * 
        (np.abs(data['volume_acceleration']) - np.abs(data['volume_deceleration']))
    )
    
    # Range Expansion Asymmetry
    data['range_expansion'] = (data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))
    data['range_contraction'] = (data['high'].shift(1) - data['low'].shift(1)) - (data['high'].shift(2) - data['low'].shift(2))
    data['range_break_score'] = (
        np.sign(data['range_expansion']) * 
        np.sign(data['range_contraction']) * 
        (np.abs(data['range_expansion']) - np.abs(data['range_contraction']))
    )
    
    # Asymmetric Range Response
    data['range_response'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['up_range_response'] = data['range_response'].where(data['range_expansion'] > 0, 0)
    data['down_range_response'] = data['range_response'].where(data['range_expansion'] < 0, 0)
    data['range_asymmetry_score'] = data['up_range_response'] - data['down_range_response']
    
    # Gap Momentum Persistence
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_follow_through'] = (data['close'] - data['open']) / data['open']
    data['gap_momentum_persistence'] = data['gap_momentum'] * data['gap_follow_through'] * np.sign(data['gap_momentum'])
    
    # Multi-Day Gap Momentum
    gap_signs = []
    gap_strengths = []
    for i in range(len(data)):
        if i >= 2:
            consecutive_count = 0
            strength_sum = 0
            for j in range(i-2, i+1):
                if j >= 1:
                    current_gap = (data['open'].iloc[j] - data['close'].iloc[j-1]) / data['close'].iloc[j-1]
                    prev_gap = (data['open'].iloc[j-1] - data['close'].iloc[j-2]) / data['close'].iloc[j-2] if j >= 2 else 0
                    if np.sign(current_gap) == np.sign(prev_gap) or j == i-2:
                        consecutive_count += 1
                        strength_sum += current_gap
            gap_signs.append(consecutive_count)
            gap_strengths.append(strength_sum)
        else:
            gap_signs.append(0)
            gap_strengths.append(0)
    
    data['consecutive_gap_count'] = gap_signs
    data['gap_momentum_strength'] = gap_strengths
    data['multi_day_gap_score'] = data['consecutive_gap_count'] * data['gap_momentum_strength']
    
    # Volume-Price Fracture Coupling
    data['volume_price_acceleration'] = (
        (data['close'] - data['close'].shift(1)) * data['volume'] - 
        (data['close'].shift(1) - data['close'].shift(2)) * data['volume'].shift(1)
    )
    data['volume_price_deceleration'] = (
        (data['close'].shift(1) - data['close'].shift(2)) * data['volume'].shift(1) - 
        (data['close'].shift(2) - data['close'].shift(3)) * data['volume'].shift(2)
    )
    data['volume_price_fracture'] = (
        np.sign(data['volume_price_acceleration']) * 
        np.sign(data['volume_price_deceleration']) * 
        (np.abs(data['volume_price_acceleration']) - np.abs(data['volume_price_deceleration']))
    )
    
    # Price-Driven Volume Fracture
    data['price_volume_correlation_break'] = (
        (data['high'] - data['low']) * data['volume'] - 
        (data['high'].shift(1) - data['low'].shift(1)) * data['volume'].shift(1)
    )
    data['price_volume_correlation_shift'] = (
        (data['high'].shift(1) - data['low'].shift(1)) * data['volume'].shift(1) - 
        (data['high'].shift(2) - data['low'].shift(2)) * data['volume'].shift(2)
    )
    data['price_volume_fracture'] = (
        np.sign(data['price_volume_correlation_break']) * 
        np.sign(data['price_volume_correlation_shift']) * 
        (np.abs(data['price_volume_correlation_break']) - np.abs(data['price_volume_correlation_shift']))
    )
    
    # Momentum Regime Transition
    data['momentum_state'] = (
        np.sign(data['close'] - data['close'].shift(3)) * 
        np.sign(data['close'].shift(3) - data['close'].shift(6))
    )
    data['momentum_strength'] = (
        np.abs(data['close'] - data['close'].shift(3)) - 
        np.abs(data['close'].shift(3) - data['close'].shift(6))
    )
    data['momentum_transition_score'] = data['momentum_state'] * data['momentum_strength']
    
    # Volume Regime Confirmation
    data['volume_regime'] = (
        np.sign(data['volume'] - data['volume'].shift(3)) * 
        np.sign(data['volume'].shift(3) - data['volume'].shift(6))
    )
    data['volume_regime_strength'] = (
        np.abs(data['volume'] - data['volume'].shift(3)) - 
        np.abs(data['volume'].shift(3) - data['volume'].shift(6))
    )
    data['volume_transition_score'] = data['volume_regime'] * data['volume_regime_strength']
    
    # Fracture Microstructure Dynamics
    data['opening_fracture'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    data['closing_fracture'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['intraday_fracture_score'] = data['opening_fracture'] * data['closing_fracture'] * np.sign(data['opening_fracture'])
    
    data['high_side_fracture'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)
    data['low_side_fracture'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)
    data['bid_ask_fracture_score'] = data['high_side_fracture'] - data['low_side_fracture']
    
    # Multi-Timeframe Fracture Integration
    data['price_volume_short_fracture'] = data['momentum_fracture_score'] * data['volume_price_fracture']
    data['range_gap_short_fracture'] = data['range_break_score'] * data['gap_momentum_persistence']
    data['short_term_fracture_alignment'] = data['price_volume_short_fracture'] * data['range_gap_short_fracture']
    
    data['momentum_volume_medium_fracture'] = data['momentum_transition_score'] * data['volume_transition_score']
    data['range_intraday_medium_fracture'] = data['range_asymmetry_score'] * data['intraday_fracture_score']
    data['medium_term_fracture_divergence'] = data['momentum_volume_medium_fracture'] - data['range_intraday_medium_fracture']
    
    # Fracture Persistence Framework
    price_fracture_counts = []
    volume_fracture_counts = []
    for i in range(len(data)):
        short_price_count = 0
        short_volume_count = 0
        medium_price_count = 0
        medium_volume_count = 0
        
        # Short-term fractures
        for j in range(max(0, i-2), i+1):
            if j >= 2:
                current_price_sign = np.sign(data['close'].iloc[j] - data['close'].iloc[j-1])
                prev_price_sign = np.sign(data['close'].iloc[j-1] - data['close'].iloc[j-2])
                if current_price_sign != prev_price_sign:
                    short_price_count += 1
                
                current_volume_sign = np.sign(data['volume'].iloc[j] - data['volume'].iloc[j-1])
                prev_volume_sign = np.sign(data['volume'].iloc[j-1] - data['volume'].iloc[j-2])
                if current_volume_sign != prev_volume_sign:
                    short_volume_count += 1
        
        # Medium-term fractures
        for j in range(max(0, i-5), i+1):
            if j >= 6:
                current_price_sign = np.sign(data['close'].iloc[j] - data['close'].iloc[j-3])
                prev_price_sign = np.sign(data['close'].iloc[j-3] - data['close'].iloc[j-6])
                if current_price_sign != prev_price_sign:
                    medium_price_count += 1
                
                current_volume_sign = np.sign(data['volume'].iloc[j] - data['volume'].iloc[j-3])
                prev_volume_sign = np.sign(data['volume'].iloc[j-3] - data['volume'].iloc[j-6])
                if current_volume_sign != prev_volume_sign:
                    medium_volume_count += 1
        
        price_fracture_counts.append(short_price_count - medium_price_count)
        volume_fracture_counts.append(short_volume_count - medium_volume_count)
    
    data['price_fracture_persistence_score'] = price_fracture_counts
    data['volume_fracture_persistence_score'] = volume_fracture_counts
    
    # Composite Fracture Alpha Construction
    data['fracture_momentum_score'] = data['momentum_fracture_score'] * data['volume_price_fracture']
    data['fracture_microstructure_score'] = data['bid_ask_fracture_score'] * data['intraday_fracture_score']
    
    data['base_fracture_alpha'] = data['fracture_momentum_score'] * data['fracture_microstructure_score']
    data['fracture_persistence_multiplier'] = data['price_fracture_persistence_score'] * data['volume_fracture_persistence_score']
    
    data['scale_enhanced_fracture_alpha'] = (
        data['base_fracture_alpha'] * 
        (1 + data['short_term_fracture_alignment']) * 
        (1 + np.abs(data['medium_term_fracture_divergence']))
    )
    
    data['fracture_alpha_core'] = data['scale_enhanced_fracture_alpha'] * data['multi_day_gap_score']
    data['final_momentum_fracture_alpha'] = data['fracture_alpha_core'] * np.sign(data['momentum_transition_score'])
    
    return data['final_momentum_fracture_alpha']
