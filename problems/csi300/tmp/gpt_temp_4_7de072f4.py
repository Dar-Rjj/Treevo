import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-period pressure calculations
        # Intraday pressure
        intraday_pressure = ((current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / 
                           (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8) * 
                           current_data['volume'].iloc[-1])
        
        # Short-term pressure
        if i >= 1:
            short_term_pressure = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / 
                                 (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8) * 
                                 current_data['volume'].iloc[-1])
        else:
            short_term_pressure = 0
        
        # Medium-term pressure
        if i >= 3:
            recent_high = current_data['high'].iloc[-3:].max()
            recent_low = current_data['low'].iloc[-3:].min()
            recent_volume_sum = current_data['volume'].iloc[-3:].sum()
            medium_term_pressure = ((current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / 
                                  (recent_high - recent_low + 1e-8) * recent_volume_sum)
        else:
            medium_term_pressure = 0
        
        # Pressure gradient
        if medium_term_pressure != 0:
            pressure_gradient = short_term_pressure / medium_term_pressure
        else:
            pressure_gradient = 0
        
        # Asymmetric elasticity measurement
        if i >= 1:
            price_change_pct = abs((current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / 
                                 (current_data['close'].iloc[-2] + 1e-8))
            range_pct = abs((current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / 
                          (current_data['close'].iloc[-1] + 1e-8))
            price_elasticity = price_change_pct / (range_pct + 1e-8)
            
            volume_elasticity = (current_data['volume'].iloc[-1] / 
                               (current_data['volume'].iloc[-2] + 1e-8)) * price_elasticity
        else:
            price_elasticity = 0
            volume_elasticity = 0
        
        # Multi-period elasticity
        if i >= 3:
            price_change_3d = abs((current_data['close'].iloc[-1] - current_data['close'].iloc[-4]) / 
                                (current_data['close'].iloc[-4] + 1e-8))
            range_3d = abs((current_data['high'].iloc[-3:].max() - current_data['low'].iloc[-3:].min()) / 
                         (current_data['close'].iloc[-1] + 1e-8))
            multi_period_elasticity = price_change_3d / (range_3d + 1e-8)
        else:
            multi_period_elasticity = 0
        
        # Gap analysis
        if i >= 1:
            gap_magnitude = abs(current_data['open'].iloc[-1] / current_data['close'].iloc[-2] - 1)
            gap_sustainability = ((current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / 
                               (abs(current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) + 1e-8))
            gap_momentum_persistence = (np.sign(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) * 
                                     np.sign(current_data['open'].iloc[-1] - current_data['close'].iloc[-2]))
        else:
            gap_magnitude = 0
            gap_sustainability = 0
            gap_momentum_persistence = 0
        
        # Intraday pressure accumulation
        buying_pressure = ((current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / 
                         (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8) * 
                         current_data['volume'].iloc[-1])
        
        selling_pressure = ((current_data['high'].iloc[-1] - current_data['close'].iloc[-1]) / 
                          (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8) * 
                          current_data['volume'].iloc[-1])
        
        # Movement efficiency
        if i >= 1:
            true_range = max(current_data['high'].iloc[-1] - current_data['low'].iloc[-1],
                           abs(current_data['high'].iloc[-1] - current_data['close'].iloc[-2]),
                           abs(current_data['low'].iloc[-1] - current_data['close'].iloc[-2]))
            movement_efficiency = (abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / 
                                (true_range + 1e-8))
            
            range_efficiency = (abs(current_data['close'].iloc[-1] - 
                                  (current_data['high'].iloc[-1] + current_data['low'].iloc[-1]) / 2) / 
                             (current_data['high'].iloc[-1] - current_data['low'].iloc[-1] + 1e-8))
        else:
            movement_efficiency = 0
            range_efficiency = 0
        
        # Volume regime analysis
        if i >= 1:
            volume_intensity = current_data['volume'].iloc[-1] / (current_data['volume'].iloc[-2] + 1e-8)
        else:
            volume_intensity = 1
        
        if i >= 3:
            volume_persistence = current_data['volume'].iloc[-1] / (current_data['volume'].iloc[-4] + 1e-8)
        else:
            volume_persistence = 1
        
        # Market regime detection
        if i >= 5:
            momentum_5d = (current_data['close'].iloc[-1] - current_data['close'].iloc[-6]) / current_data['close'].iloc[-6]
            momentum_strength = abs(momentum_5d)
            
            # Regime classification
            if momentum_strength > 0.03:  # Trending regime
                regime_weight = 0.6
            elif momentum_strength < 0.01:  # Mean-reverting regime
                regime_weight = 0.3
            else:  # Neutral regime
                regime_weight = 0.5
        else:
            regime_weight = 0.5
        
        # Combine factors with asymmetric regime-adaptive weighting
        pressure_elasticity_component = (intraday_pressure * price_elasticity + 
                                       short_term_pressure * volume_elasticity) * regime_weight
        
        gap_pressure_component = (gap_magnitude * gap_sustainability * 
                                (buying_pressure - selling_pressure)) * (1 - regime_weight)
        
        efficiency_component = (movement_efficiency * range_efficiency * 
                              multi_period_elasticity) * volume_intensity
        
        # Final factor calculation
        final_factor = (pressure_elasticity_component + gap_pressure_component + 
                       efficiency_component) * np.sign(gap_momentum_persistence)
        
        result.iloc[i] = final_factor
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
