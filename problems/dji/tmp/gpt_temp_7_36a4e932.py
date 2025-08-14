import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close Position in Range
    close_position = (df['close'] - df['low']) / intraday_range
    
    # Calculate Daily Return
    daily_return = (df['close'].pct_change()).fillna(0)
    
    # Weighted by Volume
    weighted_daily_return = daily_return * df['volume']
    
    # Combine Initial Factors
    initial_factors = close_position + weighted_daily_return
    
    # Calculate Breakout Strength
    rolling_sum_range = intraday_range.rolling(window=21).sum()
    breakout_strength = intraday_range / (rolling_sum_range / 21)
    
    # Calculate Volume Weighted Momentum
    price_change = df['close'].diff().fillna(0)
    volume_weighted_momentum = price_change * df['volume']
    
    # Combine Breakout and Intraday Factors
    combined_breakout_intraday = initial_factors + breakout_strength + volume_weighted_momentum
    
    # Calculate Short-Term Momentum
    short_term_momentum = df['close'].pct_change(7).fillna(0) + df['close'].pct_change(14).fillna(0)
    
    # Calculate Long-Term Momentum
    long_term_momentum = df['close'].pct_change(21).fillna(0) + df['close'].pct_change(63).fillna(0)
    
    # Calculate Price Momentum
    price_momentum = df['close'].pct_change(7).fillna(0) + df['close'].pct_change(21).fillna(0)
    
    # Calculate Volume Activity
    volume_7_day_avg = df['volume'].rolling(window=7).mean()
    volume_21_day_avg = df['volume'].rolling(window=21).mean()
    volume_activity = volume_7_day_avg / volume_21_day_avg
    
    # Determine Volume Impact
    volume_ratio = volume_7_day_avg / volume_21_day_avg
    adjusted_factor = combined_breakout_intraday * volume_ratio
    
    # Incorporate Momentum
    momentum_adjusted_factor = adjusted_factor + short_term_momentum + long_term_momentum
    
    # Adjust for Trend
    moving_average = df['close'].rolling(window=30).mean()
    trend_adjustment = df['close'] - moving_average
    
    # Calculate Daily High-to-Low Range
    daily_high_to_low_range = df['high'] - df['low']
    
    # Calculate Average High-to-Low Range over N Days
    n_day_avg_high_to_low_range = daily_high_to_low_range.rolling(window=21).mean()
    
    # Compare Current High-to-Low Range to N-Day Average
    high_to_low_signal = (daily_high_to_low_range > n_day_avg_high_to_low_range).astype(int)
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] / df['open']) - 1
    
    # Weight by Volume for Close-to-Open Return
    weighted_close_to_open_return = close_to_open_return * df['volume']
    
    # Measure Volume Impact
    volume_change = (df['volume'] / df['volume'].shift(1)).apply(lambda x: 0 if x == 0 else math.log(x))
    aggregate_volume_impact = volume_change.rolling(window=30).sum() / 30
    
    # Combine Weighted Close-to-Open Return and Volume Impact
    combined_close_to_open_volume = (weighted_close_to_open_return * 0.6) + (aggregate_volume_impact * 0.4)
    
    # Integrate Combined Close-to-Open and Volume Momentum with High-Low Range Signal
    integrated_component = (combined_close_to_open_volume * 0.7) + (high_to_low_signal * 0.3)
    
    # Final Alpha Factor
    final_alpha_factor = (momentum_adjusted_factor + trend_adjustment) * 0.8 + (price_momentum * 0.5) + (volume_activity * 0.7)
    final_alpha_factor = (integrated_component * 0.8) + (high_to_low_signal * 0.2)
    
    return final_alpha_factor
