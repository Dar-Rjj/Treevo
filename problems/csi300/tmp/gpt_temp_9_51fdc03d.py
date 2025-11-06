import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function for bull volume efficiency
    def calculate_bull_volume_efficiency(window_data):
        bull_volume = 0
        for i in range(len(window_data)):
            close_val = window_data['close'].iloc[i]
            open_val = window_data['open'].iloc[i]
            volume_val = window_data['volume'].iloc[i]
            if close_val > open_val:
                price_change = (close_val / open_val) - 1
                if price_change > 0:
                    bull_volume += volume_val / price_change
        return bull_volume
    
    # Helper function for bear volume efficiency
    def calculate_bear_volume_efficiency(window_data):
        bear_volume = 0
        for i in range(len(window_data)):
            close_val = window_data['close'].iloc[i]
            open_val = window_data['open'].iloc[i]
            volume_val = window_data['volume'].iloc[i]
            if close_val < open_val:
                price_change = (open_val / close_val) - 1
                if price_change > 0:
                    bear_volume += volume_val / price_change
        return bear_volume
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        current_data = data.iloc[:i+1]
        
        # Multi-Timeframe Efficiency & Momentum Calculation
        # Short-term (3-6 days)
        if i >= 6:
            short_price_eff = (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / \
                            sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                                for j in range(i-2, i+1))
            
            short_range_eff = (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / \
                            sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] 
                                for j in range(i-2, i+1))
            
            short_price_accel = ((current_data['close'].iloc[i] / current_data['close'].iloc[i-6] - 1) - 
                               (current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1))
            
            # Bull volume per return calculation
            bull_vol_current = calculate_bull_volume_efficiency(current_data.iloc[i-8:i+1])
            bull_vol_prev = calculate_bull_volume_efficiency(current_data.iloc[i-11:i-2])
            short_vol_eff_change = (bull_vol_current / bull_vol_prev - 1) if bull_vol_prev != 0 else 0
        else:
            short_price_eff = short_range_eff = short_price_accel = short_vol_eff_change = 0
        
        # Medium-term (6-12 days)
        if i >= 12:
            medium_price_eff = (current_data['close'].iloc[i] - current_data['close'].iloc[i-6]) / \
                             sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                                 for j in range(i-5, i+1))
            
            medium_range_eff = (current_data['close'].iloc[i] - current_data['close'].iloc[i-6]) / \
                             sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] 
                                 for j in range(i-5, i+1))
            
            medium_price_accel = ((current_data['close'].iloc[i] / current_data['close'].iloc[i-12] - 1) - 
                                (current_data['close'].iloc[i] / current_data['close'].iloc[i-6] - 1))
            
            bull_vol_current = calculate_bull_volume_efficiency(current_data.iloc[i-8:i+1])
            bull_vol_prev = calculate_bull_volume_efficiency(current_data.iloc[i-14:i-5])
            medium_vol_eff_change = (bull_vol_current / bull_vol_prev - 1) if bull_vol_prev != 0 else 0
        else:
            medium_price_eff = medium_range_eff = medium_price_accel = medium_vol_eff_change = 0
        
        # Long-term (10-20 days)
        if i >= 20:
            long_price_eff = (current_data['close'].iloc[i] - current_data['close'].iloc[i-10]) / \
                           sum(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]) 
                               for j in range(i-9, i+1))
            
            long_range_eff = (current_data['close'].iloc[i] - current_data['close'].iloc[i-10]) / \
                           sum(current_data['high'].iloc[j] - current_data['low'].iloc[j] 
                               for j in range(i-9, i+1))
            
            long_price_accel = ((current_data['close'].iloc[i] / current_data['close'].iloc[i-20] - 1) - 
                              (current_data['close'].iloc[i] / current_data['close'].iloc[i-10] - 1))
            
            bull_vol_current = calculate_bull_volume_efficiency(current_data.iloc[i-8:i+1])
            bull_vol_prev = calculate_bull_volume_efficiency(current_data.iloc[i-18:i-9])
            long_vol_eff_change = (bull_vol_current / bull_vol_prev - 1) if bull_vol_prev != 0 else 0
        else:
            long_price_eff = long_range_eff = long_price_accel = long_vol_eff_change = 0
        
        # Intraday Momentum Structure Integration
        high_t = current_data['high'].iloc[i]
        low_t = current_data['low'].iloc[i]
        open_t = current_data['open'].iloc[i]
        close_t = current_data['close'].iloc[i]
        volume_t = current_data['volume'].iloc[i]
        
        # Simplified session momentum (assuming volume distribution data not available)
        morning_momentum = (high_t - open_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        afternoon_momentum = (close_t - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        session_divergence = morning_momentum - afternoon_momentum
        
        # Volume distribution analysis (simplified)
        volume_skewness = (morning_momentum - afternoon_momentum)  # Simplified proxy
        
        # Bull and bear volume efficiency over 8-day window
        if i >= 8:
            bull_vol_eff = calculate_bull_volume_efficiency(current_data.iloc[i-7:i+1])
            bear_vol_eff = calculate_bear_volume_efficiency(current_data.iloc[i-7:i+1])
            bull_bear_ratio = bull_vol_eff / bear_vol_eff if bear_vol_eff != 0 else 1
        else:
            bull_bear_ratio = 1
        
        # Efficiency-weighted momentum
        short_eff_momentum = short_price_eff * session_divergence
        medium_eff_momentum = medium_price_eff * session_divergence
        long_eff_momentum = long_price_eff * session_divergence
        
        # Divergence Analysis
        short_medium_div = (short_eff_momentum - medium_eff_momentum) * volume_skewness
        medium_long_div = (medium_eff_momentum - long_eff_momentum) * volume_skewness
        cross_timeframe = (short_eff_momentum + medium_eff_momentum + long_eff_momentum) * session_divergence
        
        # Volume efficiency divergence
        vol_eff_div_score = (medium_vol_eff_change - short_vol_eff_change) * (long_vol_eff_change - medium_vol_eff_change)
        
        # Range efficiency divergence
        short_range_div = short_range_eff - medium_range_eff
        medium_range_div = medium_range_eff - long_range_eff
        
        # Intraday Strength & Liquidity Assessment
        close_position = (close_t - low_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0.5
        range_utilization = abs(close_t - open_t) / (high_t - low_t) if (high_t - low_t) > 0 else 0
        opening_gap = abs(open_t / current_data['close'].iloc[i-1] - 1) if i > 0 else 0
        
        strength_score = close_position * range_utilization * opening_gap
        
        # Liquidity filters
        if i >= 2:
            amount_ratio = current_data['amount'].iloc[i] / ((current_data['amount'].iloc[i] + 
                                                           current_data['amount'].iloc[i-1] + 
                                                           current_data['amount'].iloc[i-2]) / 3)
            
            # Volume persistence (count of last 3 days where volume > 3-day average)
            vol_persistence = 0
            for j in range(max(0, i-2), i+1):
                if j >= 3:
                    avg_vol = (current_data['volume'].iloc[j-1] + current_data['volume'].iloc[j-2] + current_data['volume'].iloc[j-3]) / 3
                    if current_data['volume'].iloc[j] > avg_vol:
                        vol_persistence += 1
            
            volume_surge = 1 if (current_data['volume'].iloc[i] > 1.3 * 
                               (current_data['volume'].iloc[i-1] + current_data['volume'].iloc[i-2] + 
                                current_data['volume'].iloc[i-3]) / 3) else 0
        else:
            amount_ratio = 1
            vol_persistence = 1
            volume_surge = 0
        
        liquidity_score = amount_ratio * vol_persistence
        
        # Volatility context
        true_range = max(high_t - low_t, 
                        abs(high_t - current_data['close'].iloc[i-1]), 
                        abs(low_t - current_data['close'].iloc[i-1])) if i > 0 else (high_t - low_t)
        
        intraday_vol_ratio = (high_t - low_t) / abs(close_t - open_t) if abs(close_t - open_t) > 0 else 1
        
        # Adaptive Alpha Generation
        # Strategy 1: High Efficiency Momentum
        if short_price_eff > 0.7 and medium_price_eff > 0.5:
            strategy1_signal = short_medium_div
            confirmation1 = 1 if (vol_eff_div_score > 0 and strength_score > 0.25) else 0
            factor1 = strategy1_signal * confirmation1 * bull_bear_ratio
        else:
            factor1 = 0
        
        # Strategy 2: Session-Enhanced Divergence
        if morning_momentum > afternoon_momentum and volume_skewness > 0:
            strategy2_signal = cross_timeframe
            confirmation2 = 1 if (range_utilization > 0.6 and volume_surge) else 0
            factor2 = strategy2_signal * confirmation2 * liquidity_score
        else:
            factor2 = 0
        
        # Strategy 3: Volatility-Adaptive Momentum
        volatility_signal = (short_medium_div + medium_long_div) * strength_score
        volume_filter = 1 if vol_eff_div_score > 0 else 0
        factor3 = volatility_signal * volume_filter * intraday_vol_ratio
        
        # Combine strategies with equal weighting
        final_factor = (factor1 + factor2 + factor3) / 3 if (factor1 != 0 or factor2 != 0 or factor3 != 0) else 0
        
        result.iloc[i] = final_factor
    
    # Fill early values with 0
    result = result.fillna(0)
    
    return result
