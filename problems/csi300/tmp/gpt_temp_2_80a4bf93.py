import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining gap behavior, range efficiency, pressure asymmetry, 
    microstructure resilience, and momentum fragility signals.
    """
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize factor values
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        current_data = data.iloc[:i+1]  # Only use current and past data
        
        # 1. Price-Gap Momentum Persistence
        if i >= 1:
            # Overnight gap calculation
            gap = (current_data['open'].iloc[i] / current_data['close'].iloc[i-1]) - 1
            gap_magnitude = abs(gap)
            
            # Gap filling behavior
            if gap != 0:
                filling_ratio = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / \
                               (current_data['close'].iloc[i-1] - current_data['open'].iloc[i])
                filling_ratio = np.clip(filling_ratio, -2, 2)  # Bound extreme values
            else:
                filling_ratio = 0
            
            # Volume confirmation during gap filling
            recent_volume_avg = current_data['volume'].iloc[max(0, i-4):i+1].mean()
            volume_ratio = current_data['volume'].iloc[i] / recent_volume_avg if recent_volume_avg > 0 else 1
            
            gap_signal = gap_magnitude * np.sign(gap) * (1 - abs(filling_ratio)) * np.log1p(volume_ratio)
        else:
            gap_signal = 0
        
        # 2. Range-Expansion Efficiency Ratio
        if i >= 1:
            # True Range calculation
            tr_current = max(
                current_data['high'].iloc[i] - current_data['low'].iloc[i],
                abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1]),
                abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1])
            )
            
            tr_prev = max(
                current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1],
                abs(current_data['high'].iloc[i-1] - current_data['close'].iloc[max(0, i-2)]),
                abs(current_data['low'].iloc[i-1] - current_data['close'].iloc[max(0, i-2)])
            )
            
            range_expansion = tr_current / tr_prev if tr_prev > 0 else 1
            
            # Price movement efficiency
            price_change = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1])
            efficiency = price_change / tr_current if tr_current > 0 else 0
            
            # Volume-range relationship
            volume_per_range = current_data['volume'].iloc[i] / tr_current if tr_current > 0 else 0
            norm_volume_range = np.log1p(volume_per_range) / np.log1p(current_data['volume'].iloc[i]) if current_data['volume'].iloc[i] > 0 else 0
            
            range_signal = range_expansion * efficiency * norm_volume_range
        else:
            range_signal = 0
        
        # 3. Pressure-Dispersion Asymmetry
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if high_low_range > 0:
            pressure = (current_data['close'].iloc[i] - (current_data['high'].iloc[i] + current_data['low'].iloc[i]) / 2) / high_low_range
            pressure = np.clip(pressure, -1, 1)
        else:
            pressure = 0
        
        # Volume-weighted pressure
        recent_pressure_avg = 0
        pressure_count = 0
        for j in range(max(0, i-2), i):
            hl_range_prev = current_data['high'].iloc[j] - current_data['low'].iloc[j]
            if hl_range_prev > 0:
                prev_pressure = (current_data['close'].iloc[j] - (current_data['high'].iloc[j] + current_data['low'].iloc[j]) / 2) / hl_range_prev
                recent_pressure_avg += prev_pressure
                pressure_count += 1
        
        if pressure_count > 0:
            recent_pressure_avg /= pressure_count
            pressure_persistence = pressure - recent_pressure_avg
        else:
            pressure_persistence = pressure
        
        volume_weight = np.log1p(current_data['volume'].iloc[i]) / 10  # Normalize volume impact
        pressure_signal = pressure_persistence * volume_weight
        
        # 4. Micro-Structure Resilience Factor
        if i >= 5:
            # Large trade detection
            volume_avg_5d = current_data['volume'].iloc[i-5:i].mean()
            volume_surge = current_data['volume'].iloc[i] / volume_avg_5d if volume_avg_5d > 0 else 1
            
            # Price impact
            typical_range = current_data['high'].iloc[i-5:i].max() - current_data['low'].iloc[i-5:i].min()
            price_impact = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / typical_range if typical_range > 0 else 0
            
            # Resilience measure (simplified)
            resilience = 1 - min(1, price_impact * volume_surge)
        else:
            resilience = 0.5
        
        resilience_signal = resilience
        
        # 5. Momentum-Fragility Divergence
        if i >= 5:
            # Multi-period momentum
            momentum_3d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1) if i >= 3 else 0
            momentum_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1) if i >= 5 else 0
            
            # Momentum consistency
            momentum_consistency = 1 if np.sign(momentum_3d) == np.sign(momentum_5d) else -1
            
            # Volume support for momentum
            volume_trend_3d = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1) if i >= 3 else 0
            volume_trend_5d = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1) if i >= 5 else 0
            
            volume_alignment = 1 if np.sign(momentum_5d) == np.sign(volume_trend_5d) else -1
            
            momentum_signal = momentum_5d * momentum_consistency * volume_alignment
        else:
            momentum_signal = 0
        
        # Combine all signals with weights
        combined_factor = (
            0.25 * gap_signal +
            0.20 * range_signal +
            0.20 * pressure_signal +
            0.15 * resilience_signal +
            0.20 * momentum_signal
        )
        
        factor_values.iloc[i] = combined_factor
    
    # Fill initial values with neutral signal
    factor_values = factor_values.fillna(0)
    
    return factor_values
