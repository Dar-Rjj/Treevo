import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Entropy Microstructure and Asymmetric Flow Integration Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1]
        
        # Core volatility asymmetry calculations
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        close_t = current_data['close'].iloc[-1]
        open_t = current_data['open'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        amount_t = current_data['amount'].iloc[-1]
        
        # Previous day data
        high_t1 = current_data['high'].iloc[-2] if i >= 1 else np.nan
        low_t1 = current_data['low'].iloc[-2] if i >= 1 else np.nan
        close_t1 = current_data['close'].iloc[-2] if i >= 1 else np.nan
        volume_t1 = current_data['volume'].iloc[-2] if i >= 1 else np.nan
        
        # Short-term volatility asymmetry
        if high_t != low_t and high_t1 != low_t1:
            short_term_vol_asym = ((high_t - close_t) - (close_t - low_t)) / (high_t - low_t) * (high_t - low_t) / (high_t1 - low_t1)
        else:
            short_term_vol_asym = 0
        
        # Medium-term volatility asymmetry (5-day window)
        if i >= 4:
            high_window = current_data['high'].iloc[-5:]
            low_window = current_data['low'].iloc[-5:]
            max_high = high_window.max()
            min_low = low_window.min()
            
            if max_high != min_low:
                medium_term_vol_asym = ((max_high - close_t) - (close_t - min_low)) / (max_high - min_low)
                vol_fractal_ratio = (high_t - low_t) / (high_t1 - low_t1) if high_t1 != low_t1 else 1
                medium_term_vol_asym *= vol_fractal_ratio
            else:
                medium_term_vol_asym = 0
        else:
            medium_term_vol_asym = 0
        
        # Directional entropy pressure
        if i >= 1:
            directional_entropy_pressure = np.sign(close_t - close_t1) * abs(close_t - close_t1) / close_t1
        else:
            directional_entropy_pressure = 0
        
        # Volatility asymmetry entropy
        if medium_term_vol_asym != 0:
            vol_asym_entropy = short_term_vol_asym / medium_term_vol_asym * directional_entropy_pressure
        else:
            vol_asym_entropy = 0
        
        # Micro-fractal efficiency
        if i >= 1 and high_t != low_t:
            micro_fractal_eff = abs(close_t - close_t1) / (high_t - low_t)
        else:
            micro_fractal_eff = 0
        
        # Volume asymmetry entropy
        if i >= 4 and high_t != low_t:
            vol_ma_5 = current_data['volume'].iloc[-5:].mean()
            vol_asym_entropy_val = (volume_t / vol_ma_5) * ((high_t - close_t) - (close_t - low_t)) / (high_t - low_t)
        else:
            vol_asym_entropy_val = 0
        
        # Gap volatility asymmetry
        if i >= 1 and abs(open_t - close_t1) > 0 and high_t != low_t:
            gap_vol_asym = (close_t - open_t) / abs(open_t - close_t1) * abs(close_t - open_t) / (high_t - low_t) * short_term_vol_asym
        else:
            gap_vol_asym = 0
        
        # Flow acceleration
        if i >= 2:
            flow_acceleration = volume_t / current_data['volume'].iloc[-3] - 1
        else:
            flow_acceleration = 0
        
        # Volatility asymmetry return
        if i >= 1:
            vol_asym_return = (close_t / close_t1 - 1) * short_term_vol_asym * directional_entropy_pressure
        else:
            vol_asym_return = 0
        
        # Volatility asymmetry persistence (3-day)
        if i >= 2:
            vol_asym_persistence = sum([
                1 if ((current_data['high'].iloc[j] - current_data['close'].iloc[j]) - 
                      (current_data['close'].iloc[j] - current_data['low'].iloc[j])) > 0 else 0 
                for j in range(-3, 0)
            ]) / 3
        else:
            vol_asym_persistence = 0.5
        
        # Volume entropy stability (3-day)
        if i >= 2:
            vol_entropy_stability = sum([
                1 if vol_asym_entropy_val > 1 else 0 
                for j in range(-3, 0)
            ]) / 3
        else:
            vol_entropy_stability = 0.5
        
        # Core microstructure components
        vol_entropy_microstructure = vol_asym_return * vol_asym_persistence * directional_entropy_pressure
        fractal_vol_entropy_eff = vol_asym_entropy_val * micro_fractal_eff * vol_entropy_stability
        
        # Gap-flow entropy
        gap_flow_entropy = gap_vol_asym * flow_acceleration * directional_entropy_pressure
        
        # Multi-scale enhancement factors
        high_asym_multiplier = 1.5 if short_term_vol_asym > 0.2 and directional_entropy_pressure > 0 else 1.0
        low_asym_multiplier = 0.7 if short_term_vol_asym < -0.2 and directional_entropy_pressure < 0 else 1.0
        vol_flow_boost = 1.3 if flow_acceleration > 0.2 and vol_asym_entropy_val > 1 else 1.0
        
        # Final alpha construction
        primary_factor = vol_entropy_microstructure * vol_asym_persistence
        secondary_factor = fractal_vol_entropy_eff * vol_entropy_stability
        tertiary_factor = gap_flow_entropy * vol_entropy_stability
        
        # Composite microstructure alpha
        composite_alpha = (primary_factor * high_asym_multiplier * low_asym_multiplier +
                          secondary_factor * vol_flow_boost +
                          tertiary_factor)
        
        result.iloc[i] = composite_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
