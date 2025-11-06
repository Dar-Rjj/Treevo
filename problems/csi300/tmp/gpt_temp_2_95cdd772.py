import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum Fracture Detection
    # Price Momentum Fracture
    short_term_break = (data['close'] - data['close'].shift(2)) - (data['close'].shift(2) - data['close'].shift(4))
    medium_term_break = (data['close'] - data['close'].shift(5)) - (data['close'].shift(5) - data['close'].shift(10))
    momentum_fracture_score = np.sign(short_term_break) * np.sign(medium_term_break) * (np.abs(short_term_break) - np.abs(medium_term_break))
    
    # Volume Momentum Fracture
    volume_acceleration = (data['volume'] - data['volume'].shift(1)) - (data['volume'].shift(1) - data['volume'].shift(2))
    volume_fracture_score = np.sign(volume_acceleration) * (np.abs(data['volume'] - data['volume'].shift(1)) - np.abs(data['volume'].shift(1) - data['volume'].shift(2)))
    
    # Range Expansion Asymmetry
    # Range Break Detection
    range_expansion = (data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1))
    range_break_score = np.sign(range_expansion) * (np.abs(data['high'] - data['low']) - np.abs(data['high'].shift(1) - data['low'].shift(1)))
    
    # Asymmetric Range Response
    up_range_response = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    range_asymmetry_score = np.where(range_expansion > 0, up_range_response, 
                                   -up_range_response)
    
    # Gap Momentum Persistence
    gap_momentum = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.001)
    gap_follow_through = (data['close'] - data['open']) / (data['open'] + 0.001)
    gap_momentum_persistence = gap_momentum * gap_follow_through * np.sign(gap_momentum)
    
    # Volume-Price Fracture Coupling
    volume_price_acceleration = ((data['close'] - data['close'].shift(1)) * data['volume']) - ((data['close'].shift(1) - data['close'].shift(2)) * data['volume'].shift(1))
    volume_price_fracture = np.sign(volume_price_acceleration) * (np.abs((data['close'] - data['close'].shift(1)) * data['volume']) - np.abs((data['close'].shift(1) - data['close'].shift(2)) * data['volume'].shift(1)))
    
    # Momentum Regime Transition
    momentum_state = np.sign(data['close'] - data['close'].shift(3)) * np.sign(data['close'].shift(3) - data['close'].shift(6))
    momentum_transition_score = momentum_state * (np.abs(data['close'] - data['close'].shift(3)) - np.abs(data['close'].shift(3) - data['close'].shift(6)))
    
    # Fracture Microstructure Dynamics
    opening_fracture = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001)
    closing_fracture = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    intraday_fracture_score = opening_fracture * closing_fracture * np.sign(opening_fracture)
    
    # Multi-Timeframe Fracture Integration
    short_term_fracture_alignment = momentum_fracture_score * volume_fracture_score * range_break_score * gap_momentum_persistence
    medium_term_fracture_divergence = momentum_transition_score * volume_price_fracture - range_asymmetry_score * intraday_fracture_score
    
    # Fracture Persistence Framework
    def count_price_fracture_persistence(data, t):
        short_term_count = 0
        for i in range(max(0, t-2), t+1):
            if i >= 2:
                sign_current = np.sign(data['close'].iloc[i] - data['close'].iloc[i-1])
                sign_prev = np.sign(data['close'].iloc[i-1] - data['close'].iloc[i-2])
                if sign_current != sign_prev:
                    short_term_count += 1
        
        medium_term_count = 0
        for i in range(max(0, t-5), t+1):
            if i >= 6:
                sign_current = np.sign(data['close'].iloc[i] - data['close'].iloc[i-3])
                sign_prev = np.sign(data['close'].iloc[i-3] - data['close'].iloc[i-6])
                if sign_current != sign_prev:
                    medium_term_count += 1
        
        return short_term_count - medium_term_count
    
    def count_volume_fracture_persistence(data, t):
        short_term_count = 0
        for i in range(max(0, t-2), t+1):
            if i >= 2:
                sign_current = np.sign(data['volume'].iloc[i] - data['volume'].iloc[i-1])
                sign_prev = np.sign(data['volume'].iloc[i-1] - data['volume'].iloc[i-2])
                if sign_current != sign_prev:
                    short_term_count += 1
        
        medium_term_count = 0
        for i in range(max(0, t-5), t+1):
            if i >= 6:
                sign_current = np.sign(data['volume'].iloc[i] - data['volume'].iloc[i-3])
                sign_prev = np.sign(data['volume'].iloc[i-3] - data['volume'].iloc[i-6])
                if sign_current != sign_prev:
                    medium_term_count += 1
        
        return short_term_count - medium_term_count
    
    price_fracture_persistence = pd.Series([count_price_fracture_persistence(data, i) for i in range(len(data))], index=data.index)
    volume_fracture_persistence = pd.Series([count_volume_fracture_persistence(data, i) for i in range(len(data))], index=data.index)
    
    # Composite Fracture Alpha Construction
    base_fracture_alpha = momentum_fracture_score * volume_price_fracture * intraday_fracture_score
    scale_enhanced_fracture_alpha = base_fracture_alpha * (1 + short_term_fracture_alignment) * (1 + np.abs(medium_term_fracture_divergence))
    final_momentum_fracture_alpha = scale_enhanced_fracture_alpha * gap_momentum_persistence * price_fracture_persistence * volume_fracture_persistence * np.sign(momentum_transition_score)
    
    return final_momentum_fracture_alpha
