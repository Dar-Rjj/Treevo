import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1]  # Only use data up to current day
        
        # Calculate Multi-timeframe Fractal Dimensions
        if i >= 8:
            # 3-day fractal
            high_3d = current_data['high'].iloc[i-2:i+1].max()
            low_3d = current_data['low'].iloc[i-2:i+1].min()
            close_diff_3d = abs(current_data['close'].iloc[i-2] - current_data['close'].iloc[i])
            fractal_3d = (high_3d - low_3d) / close_diff_3d if close_diff_3d != 0 else 1.0
            
            # 5-day fractal
            high_5d = current_data['high'].iloc[i-4:i+1].max()
            low_5d = current_data['low'].iloc[i-4:i+1].min()
            close_diff_5d = abs(current_data['close'].iloc[i-4] - current_data['close'].iloc[i])
            fractal_5d = (high_5d - low_5d) / close_diff_5d if close_diff_5d != 0 else 1.0
            
            # 8-day fractal
            high_8d = current_data['high'].iloc[i-7:i+1].max()
            low_8d = current_data['low'].iloc[i-7:i+1].min()
            close_diff_8d = abs(current_data['close'].iloc[i-7] - current_data['close'].iloc[i])
            fractal_8d = (high_8d - low_8d) / close_diff_8d if close_diff_8d != 0 else 1.0
            
            # Classify Market Regime by Fractal Complexity
            if fractal_3d > fractal_5d > fractal_8d:
                regime = 'high'
            elif fractal_8d > fractal_5d > fractal_3d:
                regime = 'low'
            else:
                regime = 'medium'
        else:
            regime = 'medium'
            fractal_3d = fractal_5d = fractal_8d = 1.0
        
        # Analyze Volume-Weighted Price Momentum
        if i >= 2:
            # Volume-Adaptive Price Changes
            volume_t = max(current_data['volume'].iloc[i], 1)
            heavy_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * np.log(volume_t)
            light_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-2]) * (1 / np.log(volume_t))
            volume_acceleration = (heavy_momentum - light_momentum) / current_data['close'].iloc[i] if current_data['close'].iloc[i] != 0 else 0
        else:
            heavy_momentum = light_momentum = volume_acceleration = 0
        
        # Multi-scale Momentum Convergence
        if i >= 10:
            sign_1d = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-1])
            sign_2d = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-2])
            sign_5d = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-5])
            sign_10d = np.sign(current_data['close'].iloc[i] - current_data['close'].iloc[i-10])
            
            short_convergence = sign_1d * sign_2d
            medium_convergence = sign_2d * sign_5d
            long_convergence = sign_5d * sign_10d
        else:
            short_convergence = medium_convergence = long_convergence = 0
        
        momentum_component = (heavy_momentum + light_momentum + volume_acceleration + 
                             short_convergence + medium_convergence + long_convergence) / 6
        
        # Assess Price-Volume Fractal Divergence
        if i >= 4:
            # Volume fractal
            volume_range = current_data['volume'].iloc[i-4:i+1].max() - current_data['volume'].iloc[i-4:i+1].min()
            volume_fractal = current_data['volume'].iloc[i] / volume_range if volume_range != 0 else 1.0
            
            # Amount fractal
            amount_range = current_data['amount'].iloc[i-4:i+1].max() - current_data['amount'].iloc[i-4:i+1].min()
            amount_fractal = current_data['amount'].iloc[i] / amount_range if amount_range != 0 else 1.0
            
            # Price-volume fractal ratio
            price_fractal_avg = (fractal_3d + fractal_5d + fractal_8d) / 3
            price_volume_ratio = price_fractal_avg / volume_fractal if volume_fractal != 0 else price_fractal_avg
            
            # Detect Fractal Divergence Signals
            if i >= 5:
                fractal_change = (fractal_3d - fractal_5d) + (fractal_5d - fractal_8d)
                volume_change = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] if current_data['volume'].iloc[i-1] != 0 else 1.0
                
                if fractal_change < 0 and volume_change > 1.2:
                    divergence_signal = -1  # Compressing fractal
                elif fractal_change > 0 and volume_change < 0.8:
                    divergence_signal = 1   # Expanding fractal
                else:
                    divergence_signal = 0   # Stable fractal
            else:
                divergence_signal = 0
        else:
            price_volume_ratio = 1.0
            divergence_signal = 0
        
        divergence_component = price_volume_ratio * divergence_signal
        
        # Compute Regime-Specific Flow Momentum
        if i >= 3:
            # Amount-Based Flow Patterns
            amount_t = current_data['amount'].iloc[i]
            amount_t1 = current_data['amount'].iloc[i-1] if current_data['amount'].iloc[i-1] != 0 else 1.0
            amount_t3 = current_data['amount'].iloc[i-3] if current_data['amount'].iloc[i-3] != 0 else 1.0
            
            large_flow = 1 if amount_t > 2 * amount_t1 else 0
            
            # Small-flow persistence (check last 3 days)
            small_flow_persistence = 1
            for j in range(3):
                if i - j >= 0 and current_data['amount'].iloc[i-j] >= 0.5 * current_data['amount'].iloc[i-j-1]:
                    small_flow_persistence = 0
                    break
            
            flow_acceleration = (amount_t - amount_t3) / amount_t3
            
            # Flow-Momentum Alignment
            if regime == 'high':
                flow_component = large_flow * flow_acceleration
            elif regime == 'low':
                flow_component = small_flow_persistence * flow_acceleration
            else:  # medium
                flow_component = (large_flow + small_flow_persistence) * flow_acceleration / 2
        else:
            flow_component = 0
        
        # Synthesize Fractal Momentum Alpha
        if regime == 'high':
            alpha = 0.5 * momentum_component + 0.3 * divergence_component + 0.2 * flow_component
        elif regime == 'medium':
            alpha = 0.4 * momentum_component + 0.4 * divergence_component + 0.2 * flow_component
        else:  # low
            alpha = 0.3 * momentum_component + 0.5 * divergence_component + 0.2 * flow_component
        
        # Apply Fractal Direction Logic
        if divergence_signal == -1 and momentum_component > 0:
            alpha *= 1.2  # Positive reinforcement
        elif divergence_signal == 1 and momentum_component < 0:
            alpha *= -1.2  # Negative reinforcement
        elif divergence_signal == 0:
            alpha *= 0.8  # Neutral dampening
        
        # Transform Output
        complexity_multiplier = {'high': 1.5, 'medium': 1.0, 'low': 0.7}[regime]
        alpha = np.tanh(alpha) * complexity_multiplier
        
        result.iloc[i] = alpha
    
    # Fill early values with 0
    result = result.fillna(0)
    return result
