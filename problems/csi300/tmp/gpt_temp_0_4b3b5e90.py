import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    for i in range(5, len(data)):
        # Current day data
        t = i
        open_t = data['open'].iloc[t]
        high_t = data['high'].iloc[t]
        low_t = data['low'].iloc[t]
        close_t = data['close'].iloc[t]
        volume_t = data['volume'].iloc[t]
        amount_t = data['amount'].iloc[t]
        
        # Previous day data
        close_t1 = data['close'].iloc[t-1]
        open_t1 = data['open'].iloc[t-1]
        volume_t1 = data['volume'].iloc[t-1]
        
        # Avoid division by zero
        def safe_div(a, b, default=0):
            return a / b if b != 0 else default
        
        # Fracture Gap Structure
        gap_open_close_prev = abs(open_t - close_t1)
        high_low_range = high_t - low_t
        close_open_diff = close_t - open_t
        
        # Gap amplitude
        gap_amplitude = safe_div(high_low_range, gap_open_close_prev, 0) * safe_div(close_open_diff, high_low_range, 0)
        
        # Gap dimension
        gap_dimension = safe_div(np.log(max(high_low_range, 1e-6)), np.log(max(gap_open_close_prev, 1e-6)), 0) * safe_div((close_t - close_t1), high_low_range, 0)
        
        # Gap persistence
        gap_persistence = np.sign(close_t - open_t) * np.sign(close_t1 - open_t1) * (volume_t - volume_t1)
        
        # Fracture Efficiency Microstructure
        # Bid fracture efficiency
        bid_fracture_eff = safe_div((close_t - low_t), high_low_range, 0) * safe_div(gap_open_close_prev, close_t1, 0) * safe_div(volume_t, amount_t, 0) * safe_div(close_open_diff, max(abs(open_t - close_t1), 1e-6), 0)
        
        # Ask fracture efficiency
        ask_fracture_eff = safe_div((high_t - close_t), high_low_range, 0) * safe_div(gap_open_close_prev, close_t1, 0) * safe_div(volume_t, amount_t, 0) * safe_div(close_open_diff, max(abs(open_t - close_t1), 1e-6), 0)
        
        # Gap closure momentum
        close_open_5 = data['close'].iloc[t] - data['close'].iloc[t-5]
        high_range_5 = data['high'].iloc[t-5:t+1].max() - data['low'].iloc[t-5:t+1].min()
        gap_closure_momentum = safe_div(close_open_diff, max(abs(open_t - close_t1), 1e-6), 0) * safe_div((volume_t - volume_t1), max(volume_t1, 1e-6), 0) * safe_div(close_open_5, max(high_range_5, 1e-6), 0)
        
        # Volume Gap Entropy Dynamics
        # Gap volume scaling
        gap_volume_scaling = safe_div(volume_t, volume_t1, 0) * safe_div(abs(close_open_diff), max(abs(open_t - close_t1), 1e-6), 0)
        
        # Gap volume field
        gap_volume_field = close_open_diff * safe_div(volume_t, high_low_range, 0) * (volume_t - volume_t1)
        
        # Gap volume entropy
        volume_window = data['volume'].iloc[t-4:t+1]
        min_vol = volume_window.min()
        max_vol = volume_window.max()
        vol_range = max_vol - min_vol
        if vol_range > 0:
            vol_norm = (volume_t - min_vol) / vol_range
            gap_volume_entropy = -vol_norm * np.log(max(vol_norm, 1e-6)) * volume_t
        else:
            gap_volume_entropy = 0
        
        # Fracture Gap Entropy Measures
        # Gap price entropy
        price_position = safe_div((close_t - low_t), high_low_range, 0)
        if 0 < price_position < 1:
            gap_price_entropy = -price_position * np.log(price_position) * safe_div(abs(close_open_diff), max(abs(open_t - close_t1), 1e-6), 0)
        else:
            gap_price_entropy = 0
        
        # Gap boundary entropy
        gap_boundary_entropy = (close_t - low_t) * (high_t - close_t) / max(high_low_range**2, 1e-6) * volume_t * gap_price_entropy
        
        # Gap convergence entropy
        gap_convergence_entropy = gap_price_entropy * gap_volume_entropy * (close_t - close_t1)
        
        # Multi-Scale Fracture Dynamics
        # High-frequency fracture
        high_freq_fracture = safe_div(close_open_diff, high_low_range, 0) * safe_div(volume_t, volume_t1, 0) * safe_div(abs(open_t - close_t1), close_t1, 0) * safe_div(abs(close_t - close_t1), high_low_range, 0)
        
        # Medium-frequency fracture
        close_diff_sum = sum(abs(data['close'].iloc[j] - data['close'].iloc[j-1]) for j in range(t-4, t+1))
        high_low_2 = data['high'].iloc[t-2] - data['low'].iloc[t-2]
        medium_freq_fracture = safe_div((close_t - data['close'].iloc[t-5]), max(close_diff_sum, 1e-6), 0) * safe_div(high_low_range, max(abs(open_t - close_t1), 1e-6), 0) * np.sign(close_t - open_t) * safe_div(high_low_range, max(high_low_2, 1e-6), 0)
        
        # Fracture divergence
        entropy_sum = gap_price_entropy + gap_volume_entropy
        fracture_divergence = safe_div(abs(gap_price_entropy - gap_volume_entropy), max(entropy_sum, 1e-6), 0) * (volume_t - volume_t1)
        
        # Fracture Transition Classification
        # High-entropy breakout
        high_entropy_breakout = gap_price_entropy > 0.6 and gap_volume_entropy > 0.5 and gap_convergence_entropy > 0
        
        # Medium-entropy trend
        medium_entropy_trend = gap_persistence > 0.5 and np.sign(open_t - close_t1) * np.sign(close_t - open_t) > 0 and gap_closure_momentum > 0.3
        
        # Low-entropy reversal
        low_entropy_reversal = gap_amplitude < -0.2 and (bid_fracture_eff - ask_fracture_eff) < -0.2 and safe_div(volume_t, volume_t1, 1) < 0.8
        
        # Hierarchical Alpha Assembly
        # Fracture structure component
        fracture_structure = 0.25 * gap_amplitude + 0.25 * gap_dimension + 0.25 * gap_persistence + 0.25 * fracture_divergence
        
        # Efficiency entropy component
        efficiency_entropy = 0.3 * bid_fracture_eff + 0.3 * ask_fracture_eff + 0.4 * gap_convergence_entropy
        
        # Multi-scale component
        multi_scale = 0.35 * high_freq_fracture + 0.35 * medium_freq_fracture + 0.3 * gap_closure_momentum
        
        # Entropy regime multiplier
        if high_entropy_breakout:
            regime_multiplier = 1.4
        elif medium_entropy_trend:
            regime_multiplier = 1.2
        elif low_entropy_reversal:
            regime_multiplier = 0.8
        else:
            regime_multiplier = 1.0
        
        # Transition adjustment
        transition_adjustment = 0
        if gap_convergence_entropy > 0.5:
            transition_adjustment += 0.2
        if fracture_divergence < -0.5:
            transition_adjustment -= 0.1
        
        # Final alpha
        final_alpha = (fracture_structure * efficiency_entropy * multi_scale) * regime_multiplier + transition_adjustment
        
        # Alpha Validation
        micro_validation = gap_boundary_entropy * gap_volume_scaling
        meso_validation = gap_convergence_entropy * gap_closure_momentum
        macro_validation = fracture_divergence * multi_scale
        
        # Apply validation weights
        validated_alpha = final_alpha * (0.4 * micro_validation + 0.4 * meso_validation + 0.2 * macro_validation)
        
        alpha.iloc[t] = validated_alpha
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
