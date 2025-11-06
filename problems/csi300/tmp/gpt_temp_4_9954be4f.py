import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate rolling windows for efficiency
    for i in range(len(data)):
        if i < 19:  # Need at least 19 periods for calculations
            alpha.iloc[i] = 0
            continue
            
        current_idx = data.index[i]
        
        # 1. Directional Volatility Sensitivity
        # Rolling 5-day high-low range (t-4 to t)
        high_5d = data['high'].iloc[i-4:i+1].max()
        low_5d = data['low'].iloc[i-4:i+1].min()
        price_range_5d = high_5d - low_5d
        
        close_change = data['close'].iloc[i] - data['close'].iloc[i-1]
        
        if price_range_5d > 0:
            if close_change > 0:
                upward_response = close_change / price_range_5d
                downward_response = 0
            elif close_change < 0:
                upward_response = 0
                downward_response = close_change / price_range_5d
            else:
                upward_response = 0
                downward_response = 0
        else:
            upward_response = 0
            downward_response = 0
        
        # 2. Order Flow Imbalance
        # Volume-Price Divergence
        volume_sum = 0
        volume_price_sum = 0
        for j in range(i-4, i+1):
            volume_j = data['volume'].iloc[j]
            volume_sum += volume_j
            volume_price_sum += volume_j * (data['close'].iloc[i] - data['open'].iloc[j])
        
        if volume_sum > 0:
            volume_price_divergence = (data['close'].iloc[i] - data['close'].iloc[i-5]) / (volume_price_sum / volume_sum)
        else:
            volume_price_divergence = 0
        
        # Tick Imbalance
        volume_sum_tick = 0
        signed_volume_sum = 0
        for j in range(i-4, i+1):
            volume_j = data['volume'].iloc[j]
            volume_sum_tick += volume_j
            if j > i-4:  # Need previous close for sign calculation
                price_change = data['close'].iloc[j] - data['close'].iloc[j-1]
                sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
                signed_volume_sum += sign * volume_j
        
        if volume_sum_tick > 0:
            tick_imbalance = signed_volume_sum / volume_sum_tick
        else:
            tick_imbalance = 0
        
        # 3. Regime-Dependent Momentum
        # Recent volatility (t-9 to t)
        recent_high = data['high'].iloc[i-9:i+1].max()
        recent_low = data['low'].iloc[i-9:i+1].min()
        recent_volatility = recent_high - recent_low
        
        # Previous volatility (t-19 to t-10)
        prev_high = data['high'].iloc[i-19:i-9].max()
        prev_low = data['low'].iloc[i-19:i-9].min()
        prev_volatility = prev_high - prev_low
        
        if recent_volatility > prev_volatility:
            # High volatility regime
            high_vol_momentum = (data['close'].iloc[i] / data['close'].iloc[i-5]) - 1
            low_vol_momentum = 0
        elif recent_volatility < prev_volatility:
            # Low volatility regime
            high_vol_momentum = 0
            low_vol_momentum = (data['close'].iloc[i] / data['close'].iloc[i-2]) - 1
        else:
            high_vol_momentum = 0
            low_vol_momentum = 0
        
        # 4. Liquidity-Adjusted Breakout
        # Breakout Quality
        current_volume = data['volume'].iloc[i]
        volume_median = data['volume'].iloc[i-9:i+1].median()
        
        if volume_median > 0:
            breakout_quality = (data['close'].iloc[i] / high_5d - 1) * (current_volume / volume_median)
        else:
            breakout_quality = 0
        
        # Support/Resistance Penetration
        # Current position in range (t-4 to t)
        current_position = (data['close'].iloc[i] - low_5d) / price_range_5d if price_range_5d > 0 else 0
        
        # Previous position in range (t-9 to t-5)
        prev_high_5d = data['high'].iloc[i-9:i-4].max()
        prev_low_5d = data['low'].iloc[i-9:i-4].min()
        prev_range_5d = prev_high_5d - prev_low_5d
        prev_position = (data['close'].iloc[i-5] - prev_low_5d) / prev_range_5d if prev_range_5d > 0 else 0
        
        support_resistance_penetration = current_position - prev_position
        
        # Combine all components
        alpha_value = (
            upward_response + downward_response +
            volume_price_divergence + tick_imbalance +
            high_vol_momentum + low_vol_momentum +
            breakout_quality + support_resistance_penetration
        )
        
        alpha.iloc[i] = alpha_value
    
    return alpha
