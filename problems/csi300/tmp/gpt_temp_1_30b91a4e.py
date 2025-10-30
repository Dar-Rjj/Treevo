import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Decay with Volume-Pressure Divergence alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 15:  # Need at least 15 days of data
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Asymmetric Momentum Decay Analysis
        # Fast Momentum Decay (5-day) - exponential decay weights
        if i >= 5:
            fast_momentum = 0
            total_weight = 0
            for j in range(6):  # t-5 to t
                weight = np.exp(-j * 0.4)  # Exponential decay
                price_ratio = current_data['close'].iloc[i-j] / current_data['close'].iloc[i-5] - 1
                fast_momentum += price_ratio * weight
                total_weight += weight
            fast_momentum /= total_weight if total_weight > 0 else 1
        else:
            fast_momentum = 0
        
        # Slow Momentum Decay (15-day) - linear weights
        if i >= 15:
            slow_momentum = 0
            for j in range(16):  # t-15 to t
                weight = 1.0  # Linear weighting
                price_ratio = current_data['close'].iloc[i-j] / current_data['close'].iloc[i-15] - 1
                slow_momentum += price_ratio * weight
            slow_momentum /= 16
        else:
            slow_momentum = 0
        
        # Momentum Decay Divergence
        momentum_divergence = fast_momentum - slow_momentum
        momentum_magnitude = max(abs(fast_momentum), abs(slow_momentum), 1e-8)
        scaled_divergence = momentum_divergence / momentum_magnitude
        
        # 2. Volume Pressure Gradient Analysis
        # Volume Pressure Index
        if i >= 1:
            volume_pressure = 0
            window_size = min(5, i)  # Use available data
            for j in range(window_size):
                idx = i - j
                prev_idx = max(0, idx - 1)
                
                # Price direction from amount (buying vs selling pressure)
                price_change = current_data['close'].iloc[idx] - current_data['close'].iloc[prev_idx]
                amount_change = current_data['amount'].iloc[idx] - current_data['amount'].iloc[prev_idx]
                
                if amount_change != 0:
                    direction = 1 if (price_change * amount_change) > 0 else -1
                    volume_weight = current_data['volume'].iloc[idx] / (current_data['volume'].iloc[i-4:i+1].mean() + 1e-8)
                    volume_pressure += direction * volume_weight
            
            volume_pressure /= window_size
        else:
            volume_pressure = 0
        
        # Volume-Momentum Pressure Divergence
        volume_momentum_divergence = scaled_divergence * volume_pressure
        
        # 3. Intraday Range Efficiency Factor
        if i >= 1:
            daily_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if daily_range > 1e-8:
                range_efficiency = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / daily_range
            else:
                range_efficiency = 0.5
        else:
            range_efficiency = 0.5
        
        # Range-Momentum Integration
        range_momentum_signal = range_efficiency * scaled_divergence
        
        # 4. Multi-Timeframe Signal Convergence
        # Short-Term Signal (3-day window)
        short_term_signal = 0
        if i >= 3:
            recent_volume_momentum = []
            for j in range(min(3, i-2)):
                idx = i - j
                if idx >= 2:
                    recent_momentum = (current_data['close'].iloc[idx] / current_data['close'].iloc[idx-2] - 1)
                    recent_volume = current_data['volume'].iloc[idx] / (current_data['volume'].iloc[idx-2:idx+1].mean() + 1e-8)
                    recent_volume_momentum.append(recent_momentum * recent_volume)
            
            if recent_volume_momentum:
                short_term_signal = np.mean(recent_volume_momentum)
        
        # Medium-Term Signal (8-day window)
        medium_term_signal = 0
        if i >= 8:
            medium_volume_momentum = []
            for j in range(min(8, i-7)):
                idx = i - j
                if idx >= 7:
                    medium_momentum = (current_data['close'].iloc[idx] / current_data['close'].iloc[idx-7] - 1)
                    medium_volume = current_data['volume'].iloc[idx] / (current_data['volume'].iloc[idx-7:idx+1].mean() + 1e-8)
                    medium_volume_momentum.append(medium_momentum * medium_volume)
            
            if medium_volume_momentum:
                medium_term_signal = np.mean(medium_volume_momentum)
        
        # Signal Convergence Scoring
        signal_alignment = 1 if (short_term_signal * medium_term_signal) > 0 else 0.3
        convergence_score = (short_term_signal * 0.6 + medium_term_signal * 0.4) * signal_alignment
        
        # 5. Adaptive Threshold Signal Generation
        # Use recent 20-day factor values for threshold calculation
        if i >= 35:  # Need enough history for thresholds
            recent_factors = []
            for j in range(min(20, i-14)):
                # Calculate simplified factor for threshold calculation
                factor_val = volume_momentum_divergence * 0.4 + range_momentum_signal * 0.3 + convergence_score * 0.3
                recent_factors.append(factor_val)
            
            if recent_factors:
                factor_std = np.std(recent_factors)
                upper_threshold = np.percentile(recent_factors, 75) if len(recent_factors) >= 4 else 0.02
                lower_threshold = np.percentile(recent_factors, 25) if len(recent_factors) >= 4 else -0.02
            else:
                factor_std = 0.01
                upper_threshold = 0.02
                lower_threshold = -0.02
        else:
            factor_std = 0.01
            upper_threshold = 0.02
            lower_threshold = -0.02
        
        # Final factor calculation
        base_factor = (volume_momentum_divergence * 0.4 + 
                      range_momentum_signal * 0.3 + 
                      convergence_score * 0.3)
        
        # Apply adaptive thresholds with volume validation
        if base_factor > upper_threshold and volume_pressure > 0:
            final_factor = min(base_factor, upper_threshold + factor_std)
        elif base_factor < lower_threshold and volume_pressure < 0:
            final_factor = max(base_factor, lower_threshold - factor_std)
        else:
            final_factor = base_factor * 0.5  # Reduce signal strength without volume confirmation
        
        result.iloc[i] = final_factor
    
    return result
