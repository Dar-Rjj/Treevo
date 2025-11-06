import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead
    for i in range(len(data)):
        if i < 20:  # Need at least 20 days for calculations
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]  # Only use data up to current day
        
        # 1. Asymmetric Range Momentum
        # Range Asymmetry
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if high_low_range > 0:
            range_asymmetry = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / high_low_range - 0.5
        else:
            range_asymmetry = 0
            
        # Range Momentum Divergence
        if i >= 10:
            asym_t = range_asymmetry
            asym_t3 = (current_data['close'].iloc[i-3] - current_data['low'].iloc[i-3]) / (current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3]) - 0.5 if (current_data['high'].iloc[i-3] - current_data['low'].iloc[i-3]) > 0 else 0
            asym_t10 = (current_data['close'].iloc[i-10] - current_data['low'].iloc[i-10]) / (current_data['high'].iloc[i-10] - current_data['low'].iloc[i-10]) - 0.5 if (current_data['high'].iloc[i-10] - current_data['low'].iloc[i-10]) > 0 else 0
            
            range_momentum_divergence = abs((asym_t - asym_t3) - (asym_t - asym_t10))
        else:
            range_momentum_divergence = 0
            
        # Volume-Weighted Asymmetry
        if i >= 4:
            avg_volume_4d = current_data['volume'].iloc[i-4:i+1].mean()
            if avg_volume_4d > 0:
                volume_weighted_asymmetry = range_asymmetry * (current_data['volume'].iloc[i] / avg_volume_4d)
            else:
                volume_weighted_asymmetry = 0
        else:
            volume_weighted_asymmetry = 0
            
        # 2. Volatility-Weighted Reversal
        # Range Price Reversal
        if high_low_range > 0:
            range_price_reversal = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / high_low_range
        else:
            range_price_reversal = 0
            
        # True Range Volatility
        tr1 = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        tr2 = abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1])
        tr3 = abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1])
        true_range = max(tr1, tr2, tr3)
        
        # Volatility-Weighted Reversal
        if i >= 4:
            avg_true_range = np.mean([max(current_data['high'].iloc[j] - current_data['low'].iloc[j], 
                                        abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1]),
                                        abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1]))
                                    for j in range(i-4, i+1)])
            if avg_true_range > 1e-8:  # Avoid division by zero
                volatility_weighted_reversal = range_price_reversal / avg_true_range
            else:
                volatility_weighted_reversal = 0
        else:
            volatility_weighted_reversal = 0
            
        # 3. Range Efficiency Metrics
        # Movement Efficiency
        if high_low_range > 0:
            movement_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range
        else:
            movement_efficiency = 0
            
        # Price Movement Efficiency
        if i >= 5:
            price_change_5d = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-5])
            sum_true_range_5d = sum([max(current_data['high'].iloc[j] - current_data['low'].iloc[j],
                                       abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1]),
                                       abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1]))
                                   for j in range(i-4, i+1)])
            if sum_true_range_5d > 0:
                price_movement_efficiency = price_change_5d / sum_true_range_5d
            else:
                price_movement_efficiency = 0
        else:
            price_movement_efficiency = 0
            
        # 4. Volume Confirmation
        # Volume Spike and Consecutive Volume Days
        if i >= 4:
            volume_window = current_data['volume'].iloc[i-4:i+1]
            volume_median = volume_window.median()
            volume_spike = current_data['volume'].iloc[i] > volume_median
            
            # Count consecutive days with volume above median
            consecutive_volume_days = 0
            for j in range(i, max(i-5, -1), -1):
                if current_data['volume'].iloc[j] > volume_median:
                    consecutive_volume_days += 1
                else:
                    break
        else:
            consecutive_volume_days = 1
            
        # 5. Momentum Alignment
        # Short-term and Medium-term Momentum
        short_term_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
        medium_term_momentum = current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1
        
        momentum_alignment = np.sign(short_term_momentum) * np.sign(medium_term_momentum) * abs(short_term_momentum - medium_term_momentum)
        
        # 6. Composite Synthesis
        # Core Factor
        core_factor = range_momentum_divergence * volume_weighted_asymmetry * volatility_weighted_reversal
        
        # Efficiency Enhanced
        efficiency_enhanced = core_factor * movement_efficiency * price_movement_efficiency
        
        # Final Factor
        final_factor = efficiency_enhanced * consecutive_volume_days * momentum_alignment
        
        result.iloc[i] = final_factor
    
    return result
