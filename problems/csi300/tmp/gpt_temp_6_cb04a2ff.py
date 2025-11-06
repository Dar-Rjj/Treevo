import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Intraday Asymmetry Compression Framework
    Generates alpha factor based on multi-timeframe fractal patterns and intraday asymmetries
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Define session boundaries (assuming 6.5 hour trading day)
    morning_end_hour = 11.5  # 11:30 AM
    afternoon_start_hour = 13.0  # 1:00 PM
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        prev_data = df.iloc[i-1] if i > 0 else current_data
        
        # Multi-Timeframe Fractal Asymmetry Analysis
        # Morning Session Components
        morning_range = current_data['high'] - current_data['low']
        morning_price_change = current_data['close'] - current_data['open']
        
        # Volatility Skew Fractal
        volatility_skew = (current_data['high'] - current_data['open']) / max(0.001, (current_data['open'] - current_data['low']))
        
        # Range Efficiency Fractal
        range_efficiency = (morning_price_change / max(0.001, morning_range)) * current_data['volume']
        
        # Microstructure Pressure
        microstructure_pressure = (current_data['open'] - prev_data['close']) / max(0.001, morning_range)
        
        # Afternoon Session Components (using current day data as proxy)
        # For simplicity, using current day high/low as afternoon session
        afternoon_range = morning_range  # Using same range as approximation
        afternoon_efficiency = abs(morning_price_change) / max(0.001, current_data['volume'] * (current_data['open'] + current_data['close']) / 2)
        
        # Range Position Shift
        morning_position = (current_data['close'] - current_data['low']) / max(0.001, morning_range)
        range_position_shift = morning_position - morning_position  # Simplified
        
        # Cross-Session Fractal Divergence
        volatility_divergence = volatility_skew - (afternoon_range / max(0.001, morning_range))
        efficiency_divergence = range_efficiency - afternoon_efficiency
        microstructure_shift = microstructure_pressure - microstructure_pressure  # Simplified
        
        # Hierarchical Fractal Compression Patterns
        # Short-Term Compression (1-5 days)
        morning_ranges = [df.iloc[i-j]['high'] - df.iloc[i-j]['low'] for j in range(1, min(6, i+1))]
        avg_morning_range = np.mean(morning_ranges) if morning_ranges else morning_range
        
        morning_compression_intensity = morning_range / max(0.001, avg_morning_range)
        afternoon_expansion_quality = afternoon_range * current_data['volume']
        cross_session_compression = morning_compression_intensity / max(0.001, afternoon_expansion_quality)
        
        # Medium-Term Fractal Context (5-20 days)
        if i >= 20:
            efficiency_patterns = []
            for j in range(1, min(21, i+1)):
                day_data = df.iloc[i-j]
                day_range = day_data['high'] - day_data['low']
                day_efficiency = (day_data['close'] - day_data['open']) / max(0.001, day_range)
                efficiency_patterns.append(day_efficiency)
            
            # Fractal Persistence (correlation of consecutive efficiency patterns)
            if len(efficiency_patterns) >= 2:
                fractal_persistence = np.corrcoef(efficiency_patterns[:-1], efficiency_patterns[1:])[0,1] if len(efficiency_patterns) >= 2 else 0
            else:
                fractal_persistence = 0
        else:
            fractal_persistence = 0
        
        # Fractal Order Flow Integration
        # Directional Volume Patterns
        up_days_volume = sum(df.iloc[i-j]['volume'] for j in range(1, min(6, i+1)) 
                           if df.iloc[i-j]['close'] > df.iloc[i-j]['open'])
        total_morning_volume = sum(df.iloc[i-j]['volume'] for j in range(1, min(6, i+1)))
        
        morning_volume_concentration = up_days_volume / max(0.001, total_morning_volume)
        
        # Price Impact Fractal Asymmetry
        morning_price_impact = abs(morning_price_change) / max(0.001, current_data['volume'] * current_data['open'])
        afternoon_price_impact = abs(morning_price_change) / max(0.001, current_data['volume'] * (current_data['open'] + current_data['close']) / 2)
        impact_divergence = morning_price_impact - afternoon_price_impact
        
        # Order Flow Fractal Persistence
        if i >= 3:
            recent_flows = []
            for j in range(1, 4):
                if i-j >= 0:
                    day_data = df.iloc[i-j]
                    flow_direction = 1 if day_data['close'] > day_data['open'] else -1
                    recent_flows.append(flow_direction * day_data['volume'])
            
            order_flow_imbalance = np.mean(recent_flows) if recent_flows else 0
            flow_reversal = abs(order_flow_imbalance - (recent_flows[-2] if len(recent_flows) >= 2 else 0))
        else:
            order_flow_imbalance = 0
            flow_reversal = 0
        
        # Fractal Volatility-Regime Processing
        # Compression Regime Classification
        high_compression = morning_compression_intensity * (1 / max(0.001, afternoon_expansion_quality))
        low_compression = (1 / max(0.001, morning_compression_intensity)) * afternoon_expansion_quality
        
        # Fractal Asymmetric Regime Signals
        high_compression_regime = volatility_divergence * cross_session_compression
        low_compression_regime = efficiency_divergence * impact_divergence
        transition_regime = order_flow_imbalance * impact_divergence
        
        # Fractal Regime Quality Assessment
        if i >= 3:
            recent_patterns = [morning_compression_intensity]  # Simplified persistence measure
            pattern_persistence = np.std(recent_patterns) if recent_patterns else 1
        else:
            pattern_persistence = 1
        
        # Core Fractal Asymmetric Factor
        primary_signal = volatility_divergence * cross_session_compression
        secondary_signal = order_flow_imbalance * (high_compression_regime + low_compression_regime + transition_regime)
        tertiary_signal = impact_divergence * morning_volume_concentration
        
        # Multi-Timeframe Fractal Confirmation
        short_term_validation = pattern_persistence * morning_volume_concentration
        medium_term_context = fractal_persistence * order_flow_imbalance
        cross_frequency_alignment = cross_session_compression * morning_compression_intensity
        
        # Final Fractal Alpha Generation
        core_factor = (primary_signal * 0.4 + secondary_signal * 0.35 + tertiary_signal * 0.25)
        confirmation_strength = (short_term_validation * 0.4 + medium_term_context * 0.35 + cross_frequency_alignment * 0.25)
        
        # Hierarchical weighting with efficiency multiplier
        fractal_efficiency_multiplier = range_efficiency * afternoon_efficiency
        
        final_alpha = core_factor * confirmation_strength * fractal_efficiency_multiplier
        
        result.iloc[i] = final_alpha if not np.isnan(final_alpha) and np.isfinite(final_alpha) else 0
    
    # Normalize the result
    if len(result) > 0:
        result = (result - result.mean()) / max(0.001, result.std())
    
    return result
