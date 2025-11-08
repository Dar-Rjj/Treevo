import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal-Momentum Decay with Adaptive Volume Entropy alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 10:  # Need sufficient history for calculations
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # 1. Fractal Momentum Structure
        # Calculate fractal high/low using rolling window
        fractal_window = 5
        if i >= fractal_window:
            fractal_high = current_data['high'].iloc[i-fractal_window+1:i+1].max()
            fractal_low = current_data['low'].iloc[i-fractal_window+1:i+1].min()
            
            if fractal_high != fractal_low:
                fractal_momentum = (current_data['close'].iloc[i] - fractal_low) / (fractal_high - fractal_low)
            else:
                fractal_momentum = 0
                
            # Fractal momentum persistence (correlation over last 5 periods)
            if i >= fractal_window + 4:
                fm_values = []
                for j in range(5):
                    start_idx = i - fractal_window - j + 1
                    end_idx = i - j + 1
                    if start_idx >= 0:
                        fh = current_data['high'].iloc[start_idx:end_idx].max()
                        fl = current_data['low'].iloc[start_idx:end_idx].min()
                        if fh != fl:
                            fm = (current_data['close'].iloc[i-j] - fl) / (fh - fl)
                            fm_values.append(fm)
                
                if len(fm_values) >= 2:
                    fm_persistence = np.corrcoef(fm_values[:-1], fm_values[1:])[0,1] if len(fm_values) >= 2 else 0
                else:
                    fm_persistence = 0
            else:
                fm_persistence = 0
                
            # Fractal momentum acceleration
            if i >= fractal_window + 2:
                start_idx1 = i - fractal_window - 2 + 1
                end_idx1 = i - 2 + 1
                start_idx2 = i - fractal_window + 1
                end_idx2 = i + 1
                
                if start_idx1 >= 0 and start_idx2 >= 0:
                    fh1 = current_data['high'].iloc[start_idx1:end_idx1].max()
                    fl1 = current_data['low'].iloc[start_idx1:end_idx1].min()
                    fh2 = current_data['high'].iloc[start_idx2:end_idx2].max()
                    fl2 = current_data['low'].iloc[start_idx2:end_idx2].min()
                    
                    if fh1 != fl1 and fh2 != fl2:
                        fm1 = (current_data['close'].iloc[i-2] - fl1) / (fh1 - fl1)
                        fm2 = (current_data['close'].iloc[i] - fl2) / (fh2 - fl2)
                        fm_acceleration = fm2 - fm1
                    else:
                        fm_acceleration = 0
                else:
                    fm_acceleration = 0
            else:
                fm_acceleration = 0
        else:
            fractal_momentum = 0
            fm_persistence = 0
            fm_acceleration = 0
        
        # 2. Momentum Decay Patterns
        if i >= fractal_window + 1 and 'fractal_momentum' in locals():
            start_idx_prev = i - fractal_window
            end_idx_prev = i
            if start_idx_prev >= 0:
                fh_prev = current_data['high'].iloc[start_idx_prev:end_idx_prev].max()
                fl_prev = current_data['low'].iloc[start_idx_prev:end_idx_prev].min()
                if fh_prev != fl_prev:
                    fm_prev = (current_data['close'].iloc[i-1] - fl_prev) / (fh_prev - fl_prev)
                    if abs(fm_prev) > 1e-6:
                        momentum_decay_rate = (fractal_momentum - fm_prev) / abs(fm_prev)
                    else:
                        momentum_decay_rate = 0
                else:
                    momentum_decay_rate = 0
            else:
                momentum_decay_rate = 0
        else:
            momentum_decay_rate = 0
        
        # 3. Adaptive Volume Entropy Analysis
        # Volume entropy over last 5 periods
        volume_window = 5
        if i >= volume_window - 1:
            volumes = current_data['volume'].iloc[i-volume_window+1:i+1].values
            total_volume = np.sum(volumes)
            
            if total_volume > 0:
                volume_ratios = volumes / total_volume
                # Remove zeros to avoid log(0)
                volume_ratios = volume_ratios[volume_ratios > 0]
                if len(volume_ratios) > 0:
                    volume_entropy = -np.sum(volume_ratios * np.log(volume_ratios))
                else:
                    volume_entropy = 0
            else:
                volume_entropy = 0
                
            # Volume concentration divergence
            if i >= volume_window:
                prev_avg_volume = current_data['volume'].iloc[i-volume_window:i].mean()
                if prev_avg_volume > 0:
                    volume_concentration = current_data['volume'].iloc[i] / prev_avg_volume
                else:
                    volume_concentration = 1
            else:
                volume_concentration = 1
        else:
            volume_entropy = 0
            volume_concentration = 1
        
        # 4. Fractal-Decay Efficiency
        if i >= fractal_window and 'fractal_momentum' in locals() and momentum_decay_rate != 0:
            price_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i])
            momentum_change = abs(momentum_decay_rate * (fractal_momentum if 'fm_prev' not in locals() else fm_prev))
            
            if momentum_change > 1e-6:
                momentum_efficiency = price_efficiency / momentum_change
            else:
                momentum_efficiency = 0
        else:
            momentum_efficiency = 0
        
        # 5. Pressure Release Dynamics
        if i >= 4:
            pressure_accumulation = 0
            for j in range(5):
                if i - j >= fractal_window:
                    start_idx = i - j - fractal_window + 1
                    end_idx = i - j + 1
                    if start_idx >= 0:
                        fh = current_data['high'].iloc[start_idx:end_idx].max()
                        fl = current_data['low'].iloc[start_idx:end_idx].min()
                        fractal_midpoint = (fh + fl) / 2
                        pressure = (current_data['close'].iloc[i-j] - fractal_midpoint) * current_data['volume'].iloc[i-j]
                        pressure_accumulation += pressure
        else:
            pressure_accumulation = 0
        
        # 6. Composite Alpha Construction
        # Core factor: Fractal Momentum Decay × Volume Entropy
        core_factor = momentum_decay_rate * volume_entropy if volume_entropy > 0 else momentum_decay_rate
        
        # Efficiency multiplier
        efficiency_multiplier = 1 + np.tanh(momentum_efficiency) if momentum_efficiency != 0 else 1
        
        # Pressure enhancement
        pressure_enhancement = 1 + np.tanh(pressure_accumulation / 1e6) if pressure_accumulation != 0 else 1
        
        # Adaptive regime filtering
        regime_filter = np.tanh(fm_persistence) * np.tanh(volume_concentration - 1)
        
        # Final alpha calculation
        alpha_value = core_factor * efficiency_multiplier * pressure_enhancement * (1 + regime_filter)
        
        # Handle edge cases
        if not np.isfinite(alpha_value):
            alpha_value = 0
            
        alpha.iloc[i] = alpha_value
    
    return alpha
