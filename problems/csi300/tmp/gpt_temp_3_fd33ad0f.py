import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Wave-Flow Pressure Dynamics
        # Directional Pressure
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        close_t = current_data['close'].iloc[-1]
        open_t = current_data['open'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        
        close_t_minus_1 = current_data['close'].iloc[-2] if i >= 1 else np.nan
        volume_t_minus_2 = current_data['volume'].iloc[-3] if i >= 2 else np.nan
        
        directional_pressure = ((high_t - close_t) - (close_t - low_t))
        if not np.isnan(close_t_minus_1):
            directional_pressure *= np.sign(close_t - close_t_minus_1)
        if not np.isnan(volume_t_minus_2) and volume_t_minus_2 != 0:
            directional_pressure *= volume_t / volume_t_minus_2
        
        # Volatility Pressure
        high_t_minus_2 = current_data['high'].iloc[-3] if i >= 2 else np.nan
        low_t_minus_2 = current_data['low'].iloc[-3] if i >= 2 else np.nan
        
        volatility_pressure = ((high_t - close_t) - (close_t - low_t))
        if not np.isnan(high_t_minus_2) and not np.isnan(low_t_minus_2) and (high_t_minus_2 - low_t_minus_2) != 0:
            volatility_pressure *= (high_t - low_t) / (high_t_minus_2 - low_t_minus_2)
        
        # Efficiency Pressure
        efficiency_pressure = ((high_t - close_t) - (close_t - low_t))
        if (high_t - low_t) != 0:
            efficiency_pressure *= abs(close_t - open_t) / (high_t - low_t)
        
        # Multi-Frequency Wave Patterns
        # Short-Term Wave
        close_window_3 = current_data['close'].iloc[-4:-1] if i >= 3 else pd.Series([np.nan])
        max_close_3 = close_window_3.max() if len(close_window_3) > 0 and not close_window_3.isna().all() else np.nan
        min_close_3 = close_window_3.min() if len(close_window_3) > 0 and not close_window_3.isna().all() else np.nan
        
        short_term_wave = ((high_t - max_close_3) - (min_close_3 - low_t)) if not np.isnan(max_close_3) and not np.isnan(min_close_3) else np.nan
        if not np.isnan(short_term_wave) and (high_t - low_t) != 0:
            short_term_wave /= (high_t - low_t)
        
        # Medium-Term Wave
        close_window_10 = current_data['close'].iloc[-11:-1] if i >= 10 else pd.Series([np.nan])
        max_close_10 = close_window_10.max() if len(close_window_10) > 0 and not close_window_10.isna().all() else np.nan
        min_close_10 = close_window_10.min() if len(close_window_10) > 0 and not close_window_10.isna().all() else np.nan
        
        medium_term_wave = ((high_t - max_close_10) - (min_close_10 - low_t)) if not np.isnan(max_close_10) and not np.isnan(min_close_10) else np.nan
        if not np.isnan(medium_term_wave) and (high_t - low_t) != 0:
            medium_term_wave /= (high_t - low_t)
        
        # Wave Alignment
        wave_alignment = short_term_wave * medium_term_wave if not np.isnan(short_term_wave) and not np.isnan(medium_term_wave) else np.nan
        
        # Regime-Specific Wave Flows
        # Volatility Regime
        high_t_minus_3 = current_data['high'].iloc[-4] if i >= 3 else np.nan
        low_t_minus_3 = current_data['low'].iloc[-4] if i >= 3 else np.nan
        
        volatility_regime = ((high_t - close_t) - (close_t - low_t))
        if not np.isnan(high_t_minus_3) and not np.isnan(low_t_minus_3) and (high_t_minus_3 - low_t_minus_3) != 0:
            volatility_regime *= (high_t - low_t) / (high_t_minus_3 - low_t_minus_3)
        
        # Volume Regime
        volume_t_minus_3 = current_data['volume'].iloc[-4] if i >= 3 else np.nan
        volume_t_minus_2 = current_data['volume'].iloc[-3] if i >= 2 else np.nan
        volume_t_minus_1 = current_data['volume'].iloc[-2] if i >= 1 else np.nan
        
        volume_regime = ((high_t - close_t) - (close_t - low_t))
        avg_volume_3 = (volume_t_minus_3 + volume_t_minus_2 + volume_t_minus_1) / 3 if not np.isnan(volume_t_minus_3) and not np.isnan(volume_t_minus_2) and not np.isnan(volume_t_minus_1) else np.nan
        if not np.isnan(avg_volume_3) and avg_volume_3 != 0:
            volume_regime *= volume_t / avg_volume_3
        
        # Trade Size Regime
        amount_t = current_data['amount'].iloc[-1]
        amount_t_minus_3 = current_data['amount'].iloc[-4] if i >= 3 else np.nan
        amount_t_minus_2 = current_data['amount'].iloc[-3] if i >= 2 else np.nan
        amount_t_minus_1 = current_data['amount'].iloc[-2] if i >= 1 else np.nan
        
        trade_size_regime = ((high_t - close_t) - (close_t - low_t))
        if volume_t != 0:
            current_trade_size = amount_t / volume_t
        else:
            current_trade_size = np.nan
            
        if not np.isnan(amount_t_minus_3) and not np.isnan(amount_t_minus_2) and not np.isnan(amount_t_minus_1):
            volume_t_minus_3 = current_data['volume'].iloc[-4] if i >= 3 else np.nan
            volume_t_minus_2 = current_data['volume'].iloc[-3] if i >= 2 else np.nan
            volume_t_minus_1 = current_data['volume'].iloc[-2] if i >= 1 else np.nan
            
            if volume_t_minus_3 != 0 and volume_t_minus_2 != 0 and volume_t_minus_1 != 0:
                avg_trade_size = ((amount_t_minus_3/volume_t_minus_3) + (amount_t_minus_2/volume_t_minus_2) + (amount_t_minus_1/volume_t_minus_1)) / 3
                if not np.isnan(current_trade_size) and avg_trade_size != 0:
                    trade_size_regime *= current_trade_size / avg_trade_size
        
        # Gap and Session Dynamics
        # Gap Wave
        close_t_minus_1 = current_data['close'].iloc[-2] if i >= 1 else np.nan
        close_t_minus_3 = current_data['close'].iloc[-4] if i >= 3 else np.nan
        
        gap_wave = (open_t - close_t_minus_1) if not np.isnan(close_t_minus_1) else np.nan
        if not np.isnan(gap_wave) and not np.isnan(close_t_minus_3) and (close_t_minus_1 - close_t_minus_3) != 0:
            gap_wave /= (close_t_minus_1 - close_t_minus_3)
        if not np.isnan(gap_wave) and not np.isnan(volume_t_minus_2) and volume_t_minus_2 != 0:
            gap_wave *= volume_t / volume_t_minus_2
        
        # Opening Wave
        opening_wave = (high_t - open_t) - (open_t - low_t)
        if (high_t - low_t) != 0:
            opening_wave *= abs(close_t - open_t) / (high_t - low_t)
        
        # Closing Wave
        closing_wave = (close_t - (high_t + low_t)/2)
        if (high_t - low_t) != 0:
            closing_wave /= (high_t - low_t)
        if not np.isnan(closing_wave) and not np.isnan(volume_t_minus_2) and volume_t_minus_2 != 0:
            closing_wave *= volume_t / volume_t_minus_2
        
        # Wave-Flow Divergence Systems
        # Efficiency Divergence
        high_t_minus_1 = current_data['high'].iloc[-2] if i >= 1 else np.nan
        low_t_minus_1 = current_data['low'].iloc[-2] if i >= 1 else np.nan
        close_t_minus_1 = current_data['close'].iloc[-2] if i >= 1 else np.nan
        open_t_minus_1 = current_data['open'].iloc[-2] if i >= 1 else np.nan
        
        current_efficiency = ((high_t - close_t) - (close_t - low_t))
        prev_efficiency = ((high_t_minus_1 - close_t_minus_1) - (close_t_minus_1 - low_t_minus_1)) if not np.isnan(high_t_minus_1) and not np.isnan(low_t_minus_1) and not np.isnan(close_t_minus_1) else np.nan
        
        efficiency_divergence = (current_efficiency - prev_efficiency) if not np.isnan(prev_efficiency) else np.nan
        current_close_open_diff = abs(close_t - open_t)
        prev_close_open_diff = abs(close_t_minus_1 - open_t_minus_1) if not np.isnan(close_t_minus_1) and not np.isnan(open_t_minus_1) else np.nan
        
        if not np.isnan(efficiency_divergence) and not np.isnan(prev_close_open_diff):
            efficiency_divergence *= (current_close_open_diff - prev_close_open_diff)
        
        # Volume Divergence
        volume_divergence = volume_t / avg_volume_3 if not np.isnan(avg_volume_3) and avg_volume_3 != 0 else np.nan
        if not np.isnan(volume_divergence):
            volume_divergence *= ((high_t - close_t) - (close_t - low_t))
        
        # Multi-Frequency Divergence
        multi_freq_divergence = wave_alignment
        
        # Composite Alpha Synthesis
        # Core Pressure Signal
        core_pressure_signal = directional_pressure * volatility_pressure if not np.isnan(directional_pressure) and not np.isnan(volatility_pressure) else np.nan
        
        # Regime Integration
        regime_integration = volatility_regime * volume_regime * trade_size_regime if not np.isnan(volatility_regime) and not np.isnan(volume_regime) and not np.isnan(trade_size_regime) else np.nan
        
        # Session Integration
        session_integration = gap_wave * opening_wave * closing_wave if not np.isnan(gap_wave) and not np.isnan(opening_wave) and not np.isnan(closing_wave) else np.nan
        
        # Wave-Flow Divergence Systems
        wave_flow_divergence = efficiency_divergence * volume_divergence * multi_freq_divergence if not np.isnan(efficiency_divergence) and not np.isnan(volume_divergence) and not np.isnan(multi_freq_divergence) else np.nan
        
        # Final Alpha
        final_alpha = 1.0
        components = [core_pressure_signal, regime_integration, session_integration, wave_flow_divergence]
        
        for component in components:
            if not np.isnan(component):
                final_alpha *= component
            else:
                final_alpha = np.nan
                break
        
        alpha.iloc[i] = final_alpha
    
    return alpha
