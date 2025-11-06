import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Extract current day data
        open_t = current_data['open'].iloc[-1]
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        close_t = current_data['close'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        
        # Previous day data
        close_t_1 = current_data['close'].iloc[-2] if i >= 1 else np.nan
        volume_t_1 = current_data['volume'].iloc[-2] if i >= 1 else np.nan
        volume_t_2 = current_data['volume'].iloc[-3] if i >= 2 else np.nan
        close_t_3 = current_data['close'].iloc[-4] if i >= 3 else np.nan
        close_t_5 = current_data['close'].iloc[-6] if i >= 5 else np.nan
        volume_t_5 = current_data['volume'].iloc[-6] if i >= 5 else np.nan
        
        # Intraday asymmetry from previous day
        if i >= 1:
            high_t_1 = current_data['high'].iloc[-2]
            low_t_1 = current_data['low'].iloc[-2]
            close_t_1_prev = current_data['close'].iloc[-2]
            intraday_asymmetry_t_1 = (high_t_1 - close_t_1_prev) - (close_t_1_prev - low_t_1)
        else:
            intraday_asymmetry_t_1 = np.nan
        
        # Range calculations
        high_t_1_prev = current_data['high'].iloc[-2] if i >= 1 else np.nan
        low_t_1_prev = current_data['low'].iloc[-2] if i >= 1 else np.nan
        
        # Skip calculation if insufficient data
        if any(pd.isna([close_t_1, volume_t_1, volume_t_2, close_t_3, close_t_5, volume_t_5, 
                        intraday_asymmetry_t_1, high_t_1_prev, low_t_1_prev])):
            result.iloc[i] = np.nan
            continue
        
        # Asymmetric Volatility Structure
        ultra_short_vol = (high_t - low_t) / close_t_1
        high_window_4 = current_data['high'].iloc[-5:-1].max() if i >= 4 else np.nan
        low_window_4 = current_data['low'].iloc[-5:-1].min() if i >= 4 else np.nan
        close_t_5_prev = current_data['close'].iloc[-6] if i >= 5 else np.nan
        short_term_vol = (high_window_4 - low_window_4) / close_t_5_prev if i >= 5 else np.nan
        volatility_asymmetry = (high_t - close_t) / (close_t - low_t) if close_t != low_t else 0
        
        # Pressure Dynamics
        opening_pressure = (open_t - low_t) / (high_t - low_t) if high_t != low_t else 0.5
        closing_pressure = (close_t - low_t) / (high_t - low_t) if high_t != low_t else 0.5
        pressure_shift = closing_pressure - opening_pressure
        
        # Momentum Stress Detection
        intraday_asymmetry = (high_t - close_t) - (close_t - low_t)
        volume_change_t = volume_t / volume_t_1 - 1
        volume_change_t_1 = volume_t_1 / volume_t_2 - 1 if i >= 2 else 0
        
        momentum_stress_gap = (
            abs(intraday_asymmetry * np.sign(volume_change_t)) - 
            abs(intraday_asymmetry_t_1 * np.sign(volume_change_t_1))
        )
        fractal_momentum_intensity = (high_t - low_t) * momentum_stress_gap * intraday_asymmetry
        
        # Efficiency Metrics
        session_efficiency = abs(close_t - open_t) / (high_t - low_t) if high_t != low_t else 0
        high_window_2 = current_data['high'].iloc[-3:-1].max() if i >= 2 else np.nan
        low_window_2 = current_data['low'].iloc[-3:-1].min() if i >= 2 else np.nan
        multi_day_efficiency = abs(close_t - close_t_3) / (high_window_2 - low_window_2) if (i >= 2 and high_window_2 != low_window_2) else 0
        volume_efficiency = (high_t - low_t) / volume_t if volume_t != 0 else 0
        
        # Volume-Price Alignment
        volume_change = volume_t / volume_t_1 - 1
        price_volume_alignment = np.sign(intraday_asymmetry) * np.sign(volume_change)
        volume_pressure_alignment = np.sign(volume_change) * np.sign(pressure_shift)
        
        # Range Dynamics
        range_change = (high_t - low_t) / (high_t_1_prev - low_t_1_prev) if (high_t_1_prev != low_t_1_prev) else 1
        high_window_4_prev = current_data['high'].iloc[-5:-1].max() if i >= 4 else np.nan
        low_window_4_prev = current_data['low'].iloc[-5:-1].min() if i >= 4 else np.nan
        range_expansion_quality = (high_t - low_t) / (high_window_4_prev - low_window_4_prev) if (i >= 4 and high_window_4_prev != low_window_4_prev) else 1
        range_consistency = np.sign(range_change - 1) * np.sign(intraday_asymmetry)
        
        # Regime Detection
        compression = (ultra_short_vol < 0.7 * short_term_vol) and (range_change < 0.7) if not pd.isna(short_term_vol) else False
        expansion = (ultra_short_vol > 1.3 * short_term_vol) and (range_change > 1.3) if not pd.isna(short_term_vol) else False
        normal = not compression and not expansion
        
        # Core Momentum Signals
        intraday_momentum = close_t / open_t - 1
        pressure_weighted_momentum = intraday_momentum * pressure_shift
        efficiency_enhanced_momentum = intraday_momentum * session_efficiency
        volatility_adjusted_momentum = intraday_momentum / ultra_short_vol if ultra_short_vol != 0 else 0
        
        # Multi-Scale Integration
        short_term_signal = momentum_stress_gap * price_volume_alignment * volume_change
        medium_term_signal = (close_t - close_t_5) * (intraday_asymmetry - intraday_asymmetry_t_1) * (volume_t / volume_t_5)
        signal_validation = np.sign(short_term_signal) * np.sign(medium_term_signal) * np.sign(volume_change)
        
        # Base Components
        pressure_efficiency = pressure_weighted_momentum * session_efficiency
        volatility_asymmetry_component = volatility_adjusted_momentum * volatility_asymmetry
        stress_aligned = fractal_momentum_intensity * price_volume_alignment
        volume_core = efficiency_enhanced_momentum * volume_pressure_alignment * volume_efficiency
        
        # Regime-Weighted Combination
        if compression:
            regime_factor = pressure_efficiency * 1.4 + stress_aligned * 1.2 + volume_core * 0.8
        elif expansion:
            regime_factor = volatility_asymmetry_component * 1.3 + stress_aligned * 1.1 + volume_core * 1.2
        else:  # normal
            regime_factor = (pressure_efficiency + volatility_asymmetry_component + stress_aligned + volume_core) / 4
        
        # Final Alpha
        final_alpha = regime_factor * range_expansion_quality * signal_validation
        
        result.iloc[i] = final_alpha
    
    return result
