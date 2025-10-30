import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Fractal Efficiency Momentum with Adaptive Breakout factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required periods
    for i in range(20, len(data)):
        if i < 25:  # Need at least 25 days for momentum calculation
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # === Fractal Efficiency Calculation ===
        
        # Price Fractal Analysis
        net_price_movement = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-20])
        
        price_path_length = 0
        for j in range(i-19, i+1):
            price_path_length += abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1])
        
        price_fractal_eff = net_price_movement / price_path_length if price_path_length > 0 else 0
        
        # Volume Fractal Analysis
        volume_change = abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-20])
        
        volume_path_length = 0
        for j in range(i-19, i+1):
            volume_path_length += abs(current_data['volume'].iloc[j] - current_data['volume'].iloc[j-1])
        
        volume_fractal_eff = volume_change / volume_path_length if volume_path_length > 0 else 0
        
        # Combined Fractal Efficiency
        volume_weighted_price_eff = price_fractal_eff * volume_fractal_eff
        
        # Efficiency Momentum (5-day change)
        if i >= 25:
            prev_eff = 0
            net_prev_price = abs(current_data['close'].iloc[i-5] - current_data['close'].iloc[i-25])
            prev_price_path = 0
            for j in range(i-24, i-4):
                prev_price_path += abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1])
            
            prev_volume_change = abs(current_data['volume'].iloc[i-5] - current_data['volume'].iloc[i-25])
            prev_volume_path = 0
            for j in range(i-24, i-4):
                prev_volume_path += abs(current_data['volume'].iloc[j] - current_data['volume'].iloc[j-1])
            
            prev_price_eff = net_prev_price / prev_price_path if prev_price_path > 0 else 0
            prev_volume_eff = prev_volume_change / prev_volume_path if prev_volume_path > 0 else 0
            prev_weighted_eff = prev_price_eff * prev_volume_eff
            
            efficiency_momentum = (volume_weighted_price_eff / prev_weighted_eff - 1) if prev_weighted_eff > 0 else 0
        else:
            efficiency_momentum = 0
        
        # === Range Breakout Analysis ===
        
        # Current Range Dynamics
        high_t = current_data['high'].iloc[i]
        low_t = current_data['low'].iloc[i]
        close_t = current_data['close'].iloc[i]
        close_t_1 = current_data['close'].iloc[i-1]
        
        true_range = max(
            high_t - low_t,
            abs(high_t - close_t_1),
            abs(low_t - close_t_1)
        )
        
        pressure_ratio = (close_t - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0.5
        
        range_efficiency = abs(close_t - close_t_1) / true_range if true_range > 0 else 0
        
        # Historical Range Context
        historical_ranges = []
        for j in range(i-20, i):
            historical_ranges.append(current_data['high'].iloc[j] - current_data['low'].iloc[j])
        
        hist_avg_range = np.mean(historical_ranges) if historical_ranges else true_range
        
        range_breakout_ratio = true_range / hist_avg_range if hist_avg_range > 0 else 1.0
        
        # Volume-Confirmed Breakout
        hist_volumes = [current_data['volume'].iloc[j] for j in range(i-20, i)]
        avg_volume = np.mean(hist_volumes) if hist_volumes else current_data['volume'].iloc[i]
        
        volume_ratio = current_data['volume'].iloc[i] / avg_volume if avg_volume > 0 else 1.0
        
        volume_pressure_score = pressure_ratio * volume_ratio
        breakout_momentum = range_breakout_ratio * volume_pressure_score
        
        # === Adaptive Regime Classification ===
        
        # Volume Fractal Regime
        if volume_fractal_eff > 0.7:
            volume_regime = 'high'
        elif volume_fractal_eff >= 0.3:
            volume_regime = 'medium'
        else:
            volume_regime = 'low'
        
        # Range Breakout Regime
        if range_breakout_ratio > 1.2:
            range_regime = 'high'
        elif range_breakout_ratio >= 0.8:
            range_regime = 'normal'
        else:
            range_regime = 'low'
        
        # === Dynamic Signal Integration ===
        
        if volume_regime == 'high' and range_regime == 'high':
            # High Volume Fractal + High Breakout
            signal = (0.6 * breakout_momentum + 
                     0.3 * efficiency_momentum + 
                     0.1 * range_efficiency)
        
        elif volume_regime == 'medium' and range_regime == 'normal':
            # Medium Volume Fractal + Normal Range
            signal = (0.4 * efficiency_momentum + 
                     0.3 * breakout_momentum + 
                     0.3 * volume_pressure_score)
        
        elif volume_regime == 'low' and range_regime == 'low':
            # Low Volume Fractal + Low Volatility
            signal = (0.7 * efficiency_momentum + 
                     0.2 * price_fractal_eff + 
                     0.1 * range_efficiency)
        
        else:
            # Default balanced approach for mixed regimes
            signal = (0.4 * efficiency_momentum + 
                     0.3 * breakout_momentum + 
                     0.2 * volume_pressure_score + 
                     0.1 * range_efficiency)
        
        result.iloc[i] = signal
    
    # Fill early values with 0
    result = result.fillna(0)
    
    return result
