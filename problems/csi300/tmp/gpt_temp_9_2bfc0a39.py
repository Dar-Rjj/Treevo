import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Short-Term Volume-Momentum Divergence with Adaptive Regime Detection
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required lookbacks
    for i in range(10, len(data)):
        current_data = data.iloc[i]
        
        # Short-Term Momentum Analysis
        # 2-Day Price Momentum
        if i >= 2:
            mom_2d = (current_data['close'] / data.iloc[i-2]['close'] - 1)
            mom_2d_dir = 1 if mom_2d > 0 else (-1 if mom_2d < 0 else 0)
        else:
            mom_2d_dir = 0
            
        # 5-Day Price Momentum
        if i >= 5:
            mom_5d = (current_data['close'] / data.iloc[i-5]['close'] - 1)
            mom_5d_dir = 1 if mom_5d > 0 else (-1 if mom_5d < 0 else 0)
        else:
            mom_5d_dir = 0
            
        # Momentum Consistency
        momentum_consistent = (mom_2d_dir == mom_5d_dir) and (mom_2d_dir != 0)
        
        # Volume Divergence Detection
        # Short-Term Volume Analysis
        vol_2d_dir = 0
        vol_5d_dir = 0
        
        if i >= 2:
            vol_2d_change = (current_data['volume'] / data.iloc[i-2]['volume'] - 1)
            vol_2d_dir = 1 if vol_2d_change > 0 else (-1 if vol_2d_change < 0 else 0)
            
        if i >= 5:
            vol_5d_change = (current_data['volume'] / data.iloc[i-5]['volume'] - 1)
            vol_5d_dir = 1 if vol_5d_change > 0 else (-1 if vol_5d_change < 0 else 0)
            
        # Volume-Momentum Divergence (using 5-day as primary)
        base_score = 0
        if mom_5d_dir != 0 and vol_5d_dir != 0:
            if mom_5d_dir < 0 and vol_5d_dir > 0:  # Bullish Divergence
                base_score = 1
            elif mom_5d_dir > 0 and vol_5d_dir < 0:  # Bearish Divergence
                base_score = -1
            elif mom_5d_dir == vol_5d_dir:  # Confirmation
                base_score = 0.5 * mom_5d_dir
            # No Signal remains 0
            
        # Adaptive Regime Detection
        # Volatility Regime
        daily_range = (current_data['high'] - current_data['low']) / current_data['close']
        
        if i >= 4:
            range_sum = 0
            for j in range(5):
                range_data = data.iloc[i-j]
                range_sum += (range_data['high'] - range_data['low']) / range_data['close']
            avg_range = range_sum / 5
            
            if daily_range > 1.4 * avg_range:
                vol_multiplier = 0.7  # High Volatility
            elif daily_range < 0.6 * avg_range:
                vol_multiplier = 1.3  # Low Volatility
            else:
                vol_multiplier = 1.0  # Normal Volatility
        else:
            vol_multiplier = 1.0
            
        # Trend Regime
        trend_multiplier = 1.0
        if i >= 10:
            trend_5d = (current_data['close'] / data.iloc[i-5]['close'] - 1)
            trend_10d = (current_data['close'] / data.iloc[i-10]['close'] - 1)
            
            if trend_5d > 0 and trend_10d > 0:  # Uptrend
                trend_multiplier = 1.2 if base_score > 0 else 0.8
            elif trend_5d < 0 and trend_10d < 0:  # Downtrend
                trend_multiplier = 1.2 if base_score < 0 else 0.8
            # Sideways remains 1.0
            
        # Volume Regime
        if i >= 4:
            vol_sum = sum(data.iloc[i-j]['volume'] for j in range(5))
            avg_volume = vol_sum / 5
            
            if current_data['volume'] > 2 * avg_volume:
                volume_multiplier = 1.5  # Volume Spike
            elif current_data['volume'] < 0.5 * avg_volume:
                volume_multiplier = 0.5  # Volume Drought
            else:
                volume_multiplier = 1.0
        else:
            volume_multiplier = 1.0
            
        # Timeframe Consistency Bonus
        consistency_multiplier = 1.0
        
        # Check if both 2-day and 5-day show same divergence
        if (mom_2d_dir != 0 and vol_2d_dir != 0 and 
            mom_5d_dir != 0 and vol_5d_dir != 0):
            if ((mom_2d_dir < 0 and vol_2d_dir > 0 and mom_5d_dir < 0 and vol_5d_dir > 0) or
                (mom_2d_dir > 0 and vol_2d_dir < 0 and mom_5d_dir > 0 and vol_5d_dir < 0)):
                consistency_multiplier *= 1.5
                
        # Momentum consistent across timeframes
        if momentum_consistent:
            consistency_multiplier *= 1.2
            
        # Multiple regime confirmations
        regime_confirmations = 0
        if vol_multiplier != 1.0:
            regime_confirmations += 1
        if trend_multiplier != 1.0:
            regime_confirmations += 1
        if volume_multiplier != 1.0:
            regime_confirmations += 1
            
        if regime_confirmations >= 2:
            consistency_multiplier *= 1.1
            
        # Calculate final factor value
        factor_value = (base_score * vol_multiplier * trend_multiplier * 
                       volume_multiplier * consistency_multiplier)
        
        factor.iloc[i] = factor_value
        
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
