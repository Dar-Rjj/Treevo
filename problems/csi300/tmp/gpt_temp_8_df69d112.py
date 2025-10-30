import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days of data
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Multi-Timeframe Momentum Efficiency
        # Short-term efficiency
        if i >= 3:
            short_price_change = current_data['close'].iloc[i] - current_data['close'].iloc[i-3]
            short_volatility_sum = sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] for j in range(i-2, i+1))
            short_efficiency = short_price_change / short_volatility_sum if short_volatility_sum != 0 else 0
        else:
            short_efficiency = 0
            
        # Medium-term efficiency
        if i >= 10:
            medium_price_change = current_data['close'].iloc[i] - current_data['close'].iloc[i-10]
            medium_volatility_sum = sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] for j in range(i-9, i+1))
            medium_efficiency = medium_price_change / medium_volatility_sum if medium_volatility_sum != 0 else 0
        else:
            medium_efficiency = 0
            
        # Long-term efficiency
        long_price_change = current_data['close'].iloc[i] - current_data['close'].iloc[i-20]
        long_volatility_sum = sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] for j in range(i-19, i+1))
        long_efficiency = long_price_change / long_volatility_sum if long_volatility_sum != 0 else 0
        
        # Momentum acceleration and core
        momentum_acceleration = (short_efficiency - medium_efficiency) * (medium_efficiency - long_efficiency)
        momentum_core = short_efficiency * medium_efficiency * long_efficiency
        
        # Volume-Price Divergence Quality
        if i >= 10:
            # Price trend
            price_trend = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
            
            # Volume trend
            volume_trend = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
            
            # Divergence score
            divergence_score = price_trend - volume_trend
            
            # Efficiency divergence
            short_efficiency_5d = (current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) / sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] for j in range(i-4, i+1)) if i >= 5 else 0
            medium_efficiency_10d = medium_efficiency  # Already calculated
            
            efficiency_divergence = short_efficiency_5d - medium_efficiency_10d
            
            # Divergence quality
            divergence_quality = divergence_score * efficiency_divergence * abs(price_trend)
        else:
            divergence_quality = 0
            divergence_score = 0
            
        # Intraday Breakout Pattern
        # Morning strength
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        morning_strength = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range if high_low_range != 0 else 0
        
        # Afternoon reversal
        hl2 = (current_data['high'].iloc[i] + current_data['low'].iloc[i]) / 2
        afternoon_reversal = (current_data['close'].iloc[i] - hl2) / high_low_range if high_low_range != 0 else 0
        
        # Resistance break
        resistance_break = 1 if current_data['high'].iloc[i] > max(current_data['high'].iloc[i-20:i]) else 0
        
        # Support break
        support_break = 1 if current_data['low'].iloc[i] < min(current_data['low'].iloc[i-20:i]) else 0
        
        # Breakout pattern
        breakout_pattern = (resistance_break - support_break) * morning_strength * afternoon_reversal
        
        # Volume-Volatility Validation
        if i >= 5:
            # Volume acceleration
            volume_acceleration = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
            
            # Volume efficiency
            recent_volumes = [current_data['volume'].iloc[j] for j in range(i-4, i+1)]
            volume_efficiency = current_data['volume'].iloc[i] / min(recent_volumes) if min(recent_volumes) != 0 else 0
            
            # Recent volatility
            recent_volatility = high_low_range / current_data['close'].iloc[i] if current_data['close'].iloc[i] != 0 else 0
            
            # Historical volatility
            hist_volatility_sum = sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] for j in range(i-9, i+1))
            hist_close_sum = sum(current_data['close'].iloc[j] for j in range(i-9, i+1))
            historical_volatility = hist_volatility_sum / hist_close_sum if hist_close_sum != 0 else 0
            
            # Volume-volatility score
            volume_volatility_score = volume_acceleration * volume_efficiency * (recent_volatility / historical_volatility if historical_volatility != 0 else 0)
        else:
            volume_volatility_score = 0
            volume_efficiency = 0
            
        # Composite Factor Integration
        momentum_component = momentum_core * momentum_acceleration
        divergence_component = divergence_quality * abs(divergence_score)
        pattern_component = breakout_pattern * 2
        validation_component = volume_volatility_score * volume_efficiency
        
        core_factor = momentum_component * divergence_component * pattern_component * validation_component
        
        # Adaptive Signal Enhancement
        # True range
        true_range_1 = high_low_range
        true_range_2 = abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1])
        true_range_3 = abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1])
        true_range = max(true_range_1, true_range_2, true_range_3)
        
        # Volatility regime
        recent_true_ranges = [max(current_data['high'].iloc[j] - current_data['low'].iloc[j], 
                                 abs(current_data['high'].iloc[j] - current_data['close'].iloc[j-1]),
                                 abs(current_data['low'].iloc[j] - current_data['close'].iloc[j-1])) for j in range(i-19, i+1)]
        volatility_regime = true_range / (sum(recent_true_ranges) / 20) if sum(recent_true_ranges) != 0 else 1
        
        # Volume regime
        recent_volumes_20d = [current_data['volume'].iloc[j] for j in range(i-19, i+1)]
        volume_regime = current_data['volume'].iloc[i] / (sum(recent_volumes_20d) / 20) if sum(recent_volumes_20d) != 0 else 1
        
        # Volume efficiency momentum
        if i >= 5:
            volume_efficiency_momentum = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1) / (high_low_range / current_data['close'].iloc[i]) if (high_low_range / current_data['close'].iloc[i]) != 0 else 0
        else:
            volume_efficiency_momentum = 0
            
        # Final factor
        final_factor = core_factor / (1 + volatility_regime) * volume_regime * volume_efficiency_momentum
        
        result.iloc[i] = final_factor
    
    return result
