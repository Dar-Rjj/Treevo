import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Price-Volume Divergence Efficiency Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate components for each day
    for i in range(1, len(df)):
        if i < 1:
            continue
            
        # Current day data
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        open_t = df['open'].iloc[i]
        volume_t = df['volume'].iloc[i]
        
        # Previous day data
        close_t_minus_1 = df['close'].iloc[i-1]
        volume_t_minus_1 = df['volume'].iloc[i-1]
        
        # 1. Calculate True Range
        tr1 = high_t - low_t
        tr2 = abs(high_t - close_t_minus_1)
        tr3 = abs(low_t - close_t_minus_1)
        true_range = max(tr1, tr2, tr3)
        
        # 2. Calculate Close Position in Range
        if (high_t - low_t) > 0:
            close_position = (close_t - low_t) / (high_t - low_t)
        else:
            close_position = 0.5
        
        # 3. Calculate Intraday Return Efficiency
        intraday_return = (close_t - open_t) / open_t if open_t != 0 else 0
        if (high_t - low_t) > 0:
            price_path_efficiency = abs(close_t - open_t) / (high_t - low_t)
        else:
            price_path_efficiency = 0
        
        # 4. Calculate Volume Momentum Profile
        if volume_t_minus_1 > 0:
            volume_acceleration = (volume_t - volume_t_minus_1) / volume_t_minus_1
        else:
            volume_acceleration = 0
            
        if (high_t - low_t) > 0:
            volume_concentration = volume_t / (high_t - low_t)
        else:
            volume_concentration = 0
        
        # 5. Calculate Volume-Weighted Extremes
        if (high_t - low_t) > 0:
            upside_volume_efficiency = ((close_t - low_t) / (high_t - low_t)) * volume_t
            downside_volume_efficiency = ((high_t - close_t) / (high_t - low_t)) * volume_t
            net_volume_bias = upside_volume_efficiency - downside_volume_efficiency
        else:
            upside_volume_efficiency = 0
            downside_volume_efficiency = 0
            net_volume_bias = 0
        
        # 6. Compute Multi-timeframe Momentum
        if close_t_minus_1 > 0:
            short_term_momentum = close_t / close_t_minus_1 - 1
        else:
            short_term_momentum = 0
            
        if (high_t - low_t) > 0:
            intraday_momentum = (close_t - open_t) / (high_t - low_t)
        else:
            intraday_momentum = 0
        
        # 7. Calculate Divergence Signals
        momentum_divergence = short_term_momentum * intraday_momentum
        close_to_extreme_distance = min(close_t - low_t, high_t - close_t)
        
        # 8. Combine Volume and Price Extremes
        volume_weighted_reversal = close_to_extreme_distance * volume_acceleration
        asymmetric_volume_impact = net_volume_bias * close_position
        
        # 9. Calculate Historical Context (using rolling window of 20 days)
        if i >= 20:
            # Calculate average daily range
            daily_ranges = []
            for j in range(i-19, i+1):
                if j >= 0:
                    daily_ranges.append(df['high'].iloc[j] - df['low'].iloc[j])
            avg_daily_range = np.mean(daily_ranges) if daily_ranges else 1
        else:
            avg_daily_range = 1
        
        # 10. Compute Reversal Strength Confirmation
        volume_confirmed_divergence = volume_weighted_reversal * abs(momentum_divergence)
        if avg_daily_range > 0:
            range_context = volume_confirmed_divergence / avg_daily_range
        else:
            range_context = volume_confirmed_divergence
        
        # 11. Final Factor Construction
        # Combine all components with efficiency weighting
        if avg_daily_range > 0:
            factor_value = (asymmetric_volume_impact * range_context * 
                           price_path_efficiency / avg_daily_range)
        else:
            factor_value = asymmetric_volume_impact * range_context * price_path_efficiency
        
        result.iloc[i] = factor_value
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
