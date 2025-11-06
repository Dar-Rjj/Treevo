import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple technical indicators:
    - Volatility-adjusted momentum
    - Volume-weighted price efficiency
    - Cumulative breakout momentum
    - Gap-fill trend continuation
    - Volume-confirmed support bounce
    - Amount-based price efficiency
    - Multi-timeframe momentum convergence
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        # Multi-Day Volatility-Adjusted Momentum
        close_10 = df['close'].iloc[i-10]
        close_current = df['close'].iloc[i]
        ten_day_return = (close_current - close_10) / close_10 if close_10 != 0 else 0
        
        recent_closes = df['close'].iloc[i-9:i+1]
        ten_day_volatility = recent_closes.std() if len(recent_closes) > 1 else 0
        volatility_momentum = ten_day_return / ten_day_volatility if ten_day_volatility != 0 else 0
        
        # Volume-Weighted Price Efficiency
        cumulative_range = 0
        volume_weighted_change = 0
        for j in range(5):
            if i-j-1 >= 0:
                high_low_range = df['high'].iloc[i-j] - df['low'].iloc[i-j]
                cumulative_range += high_low_range
                
                price_change = df['close'].iloc[i-j] - df['close'].iloc[i-j-1]
                volume_weighted_change += price_change * df['volume'].iloc[i-j]
        
        volume_efficiency = volume_weighted_change / cumulative_range if cumulative_range != 0 else 0
        
        # Cumulative Breakout Momentum
        twenty_day_high = df['high'].iloc[i-19:i+1].max()
        cumulative_breakout = 0
        for j in range(5):
            if i-j >= 0:
                breakout_strength = (df['close'].iloc[i-j] - twenty_day_high) / twenty_day_high
                cumulative_breakout += breakout_strength * df['volume'].iloc[i-j]
        
        twenty_day_volatility = df['close'].iloc[i-19:i+1].std() if len(df['close'].iloc[i-19:i+1]) > 1 else 0
        adjusted_breakout = cumulative_breakout / twenty_day_volatility if twenty_day_volatility != 0 else 0
        
        # Gap-Fill Trend Continuation
        gap_accumulation = 0
        range_total = 0
        for j in range(3):
            if i-j-1 >= 0:
                gap = abs(df['open'].iloc[i-j] - df['close'].iloc[i-j-1])
                gap_accumulation += gap
                
                daily_range = df['high'].iloc[i-j] - df['low'].iloc[i-j]
                range_total += daily_range
        
        gap_fill_momentum = gap_accumulation / range_total if range_total != 0 else 0
        
        # Volume-Confirmed Support Bounce
        ten_day_low = df['low'].iloc[i-9:i+1].min()
        volume_weighted_bounce = 0
        for j in range(5):
            if i-j >= 0:
                bounce = df['close'].iloc[i-j] - ten_day_low
                volume_weighted_bounce += bounce * df['volume'].iloc[i-j]
        
        current_range = df['high'].iloc[i] - df['low'].iloc[i]
        normalized_bounce = volume_weighted_bounce / current_range if current_range != 0 else 0
        
        # Amount-Based Price Efficiency
        cumulative_amount = 0
        price_movement = 0
        for j in range(5):
            if i-j-1 >= 0:
                cumulative_amount += df['amount'].iloc[i-j]
                price_change_abs = abs(df['close'].iloc[i-j] - df['close'].iloc[i-j-1])
                price_movement += price_change_abs
        
        amount_efficiency = price_movement / cumulative_amount if cumulative_amount != 0 else 0
        
        # Multi-Timeframe Momentum Convergence
        close_5 = df['close'].iloc[i-5] if i-5 >= 0 else df['close'].iloc[i]
        close_20 = df['close'].iloc[i-20] if i-20 >= 0 else df['close'].iloc[i]
        
        short_momentum = (close_current - close_5) / close_5 if close_5 != 0 else 0
        medium_momentum = (close_current - close_10) / close_10 if close_10 != 0 else 0
        long_momentum = (close_current - close_20) / close_20 if close_20 != 0 else 0
        
        momentum_convergence = short_momentum + medium_momentum + long_momentum
        
        # Combine all components with equal weighting
        combined_factor = (
            volatility_momentum +
            volume_efficiency +
            adjusted_breakout +
            gap_fill_momentum +
            normalized_bounce +
            amount_efficiency +
            momentum_convergence
        ) / 7.0
        
        result.iloc[i] = combined_factor
    
    return result
