import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required rolling windows
    for i in range(len(data)):
        if i < 10:  # Need at least 10 periods for calculations
            result.iloc[i] = 0
            continue
            
        # Extract current and historical data
        current = data.iloc[i]
        prev_data = data.iloc[max(0, i-10):i+1]
        
        # Gap-Based Reversal Detection
        # Gap Efficiency Analysis
        if i >= 2:
            short_term_range = prev_data['high'].iloc[-3:].max() - prev_data['low'].iloc[-3:].min()
            short_gap_eff = (current['open'] - prev_data['close'].iloc[-2]) / (short_term_range + 1e-8)
        else:
            short_gap_eff = 0
            
        if i >= 4:
            medium_term_range = prev_data['high'].iloc[-5:].max() - prev_data['low'].iloc[-5:].min()
            medium_gap_eff = (current['open'] - prev_data['close'].iloc[-2]) / (medium_term_range + 1e-8)
        else:
            medium_gap_eff = 0
            
        gap_efficiency_divergence = short_gap_eff - medium_gap_eff
        
        # Gap Reversal Patterns
        gap_exhaustion = (current['close'] - current['open']) * (current['open'] - prev_data['close'].iloc[-2])
        gap_direction_vs_close = np.sign(current['open'] - prev_data['close'].iloc[-2]) * np.sign(current['close'] - current['open'])
        
        # Multi-timeframe Gap Context
        if i >= 5:
            recent_high_low_range = prev_data['high'].iloc[-6:].max() - prev_data['low'].iloc[-6:].min()
            gap_proximity = (current['open'] - prev_data['low'].iloc[-6:].min()) / (recent_high_low_range + 1e-8)
            
            if i >= 6:
                prev_high_max = prev_data['high'].iloc[-7:-1].max()
                prev_prev_high_max = prev_data['high'].iloc[-8:-2].max() if i >= 7 else prev_high_max
                gap_breakout_failure = 1 if (current['open'] > prev_prev_high_max and current['close'] < prev_high_max) else 0
            else:
                gap_breakout_failure = 0
        else:
            gap_proximity = 0
            gap_breakout_failure = 0
        
        # Fractal Momentum Exhaustion
        # Movement Efficiency Analysis
        daily_movement_eff = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
        
        # Fractal Efficiency
        price_changes = [abs(prev_data['close'].iloc[j] - prev_data['close'].iloc[j-1]) for j in range(1, len(prev_data))]
        fractal_eff = abs(current['close'] - prev_data['close'].iloc[-11]) / (sum(price_changes[-9:]) + 1e-8)
        
        efficiency_divergence = daily_movement_eff - fractal_eff
        
        # Momentum Fracture Detection
        if i >= 2:
            price_acceleration = (current['close'] - prev_data['close'].iloc[-2]) / (prev_data['close'].iloc[-2] - prev_data['close'].iloc[-3] + 1e-8)
        else:
            price_acceleration = 0
            
        if i >= 5:
            multi_scale_fracture = 1 if (abs(price_acceleration) > 2.0 and 
                                        np.sign(current['close'] - prev_data['close'].iloc[-2]) != 
                                        np.sign(current['close'] - prev_data['close'].iloc[-6])) else 0
            fracture_direction = np.sign(current['close'] - prev_data['close'].iloc[-2]) * np.sign(current['close'] - prev_data['close'].iloc[-6])
        else:
            multi_scale_fracture = 0
            fracture_direction = 0
        
        # Efficiency Reversal Signals
        over_extended_movement = daily_movement_eff * price_acceleration
        fractal_exhaustion = efficiency_divergence * fracture_direction
        
        # Volume-Volatility Exhaustion
        # Volume Spike Analysis
        if i >= 10:
            avg_volume_10 = prev_data['volume'].iloc[-11:-1].mean()
            abnormal_volume = current['volume'] / (avg_volume_10 + 1e-8)
            
            avg_volume_5 = prev_data['volume'].iloc[-6:-1].mean()
            volume_climax = 1 if current['volume'] > 2 * avg_volume_5 else 0
            
            if i >= 6:
                avg_volume_prev = prev_data['volume'].iloc[-7:-2].mean()
                volume_exhaustion = 1 if (current['volume'] < 0.7 * prev_data['volume'].iloc[-2] and 
                                        prev_data['volume'].iloc[-2] > 1.5 * avg_volume_prev) else 0
            else:
                volume_exhaustion = 0
        else:
            abnormal_volume = 0
            volume_climax = 0
            volume_exhaustion = 0
        
        # Volume-Volatility Integration
        true_range = max(current['high'] - current['low'], 
                        abs(current['high'] - prev_data['close'].iloc[-2]), 
                        abs(current['low'] - prev_data['close'].iloc[-2]))
        
        if i >= 2:
            volume_range_ratio = current['volume'] / (prev_data['volume'].iloc[-2] + prev_data['volume'].iloc[-3] + 1e-8)
        else:
            volume_range_ratio = 0
            
        range_flow = current['amount'] * (current['close'] - current['open']) / (true_range + 1e-8)
        
        # Volume-Price Divergence
        if i >= 5:
            avg_volume_5_prev = prev_data['volume'].iloc[-6:-1].mean()
            price_change_pct = abs(current['close'] - prev_data['close'].iloc[-2]) / (prev_data['close'].iloc[-2] + 1e-8)
            high_volume_low_move = (current['volume'] / (avg_volume_5_prev + 1e-8)) * price_change_pct
            
            volume_trend_vs_price = np.sign(current['close'] - prev_data['close'].iloc[-6]) * np.sign(current['volume'] - prev_data['volume'].iloc[-6])
        else:
            high_volume_low_move = 0
            volume_trend_vs_price = 0
        
        # Multi-Timeframe Integration
        # Gap-Fractal Alignment
        gap_fractal_alignment_1 = gap_efficiency_divergence * efficiency_divergence
        gap_fractal_alignment_2 = gap_exhaustion * multi_scale_fracture
        gap_fractal_alignment_3 = gap_proximity * fractal_exhaustion
        
        # Volume Confirmation
        volume_confirmation_1 = abnormal_volume * range_flow
        volume_confirmation_2 = volume_exhaustion * high_volume_low_move
        volume_confirmation_3 = volume_range_ratio * volume_trend_vs_price
        
        # Timeframe Weighting
        short_term_signals = gap_exhaustion * abnormal_volume
        medium_term_signals = gap_efficiency_divergence * high_volume_low_move
        long_term_context = gap_proximity * fractal_eff
        
        # Composite Factor Construction
        # Core Reversal Components
        gap_fractal_reversal = gap_efficiency_divergence * efficiency_divergence * multi_scale_fracture
        volume_exhaustion_core = volume_exhaustion * high_volume_low_move * range_flow
        
        # Multi-timeframe Enhancement
        short_term_weighting = short_term_signals * gap_fractal_reversal
        medium_term_weighting = medium_term_signals * volume_exhaustion_core
        context_adjustment = long_term_context * multi_scale_fracture
        
        # Final Factor Integration
        primary_factor = short_term_weighting * medium_term_weighting * context_adjustment
        volatility_normalized = primary_factor / (true_range + 1e-8)
        
        result.iloc[i] = volatility_normalized
    
    return result
