import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday High-to-Low Return
    high_to_low_return = (df['high'] - df['low']) / df['low']
    
    # Calculate Intraday Open-to-Close Return
    open_to_close_return = (df['close'] - df['open']) / df['open']
    
    # Combine Intraday Returns
    combined_intraday_return = (high_to_low_return + open_to_close_return) / 2
    
    # Weight by Volume and Amount
    volume_weighted_return = combined_intraday_return * df['volume']
    amount_weighted_return = combined_intraday_return * df['amount']
    
    # Calculate Daily Price Range
    daily_price_range = df['high'] - df['low']
    
    # Cumulative Sum of Ranges
    cumulative_range_sum = daily_price_range.rolling(window=10).sum()
    
    # Enhanced Integrated Price Change
    daily_price_change = df['close'].diff()
    high_to_low_ratio = df['high'] / df['low']
    enhanced_integrated_price_change = daily_price_change * high_to_low_ratio
    
    # Enhanced Momentum Indicator
    momentum_indicator = (df['close'] / df['close'].shift(1)) * cumulative_range_sum
    close_rolling_mean = df['close'].rolling(window=5).mean().shift(1)
    momentum_indicator += close_rolling_mean
    
    # Calculate Momentum Score
    last_5_days_price_changes = daily_price_change.rolling(window=5).sum()
    close_10_day_mean = df['close'].rolling(window=10).mean()
    momentum_score = last_5_days_price_changes - close_10_day_mean
    
    # Integrate Price and Volume Changes
    price_change = df['close'].diff()
    volume_change = df['volume'].diff()
    integrated_pv_changes = (price_change * volume_change).rolling(window=5).sum()
    
    # Smooth the Indicator
    smoothed_indicator = momentum_indicator.rolling(window=7).mean()
    
    # Synthesize Indicators
    synthesized_indicator = smoothed_indicator + (volume_weighted_return + amount_weighted_return) / 2
    
    # Adjust for VWAP
    vwap = (df[['high', 'low', 'close']].mean(axis=1) * df['volume']).cumsum() / df['volume'].cumsum()
    adjusted_synthesized_indicator = synthesized_indicator / vwap
    
    # Calculate Intraday Momentum Factor
    high_low_range = df['high'] - df['low']
    high_close_ratio = df['high'] / df['close']
    
    # Calculate Volume Reversal Factor
    volume_difference = df['volume'].diff()
    volume_reversal_factor = volume_difference.sign()
    
    # Combine Momentum and Reversal Factors
    combined_momentum = (enhanced_integrated_price_change * volume_reversal_factor)
    
    # Final Alpha Factor
    final_alpha_factor = momentum_indicator * momentum_score
    final_alpha_factor = final_alpha_factor / vwap
    final_alpha_factor = final_alpha_factor * combined_momentum
    ema_alpha = 0.1
    ema_close = df['close'].ewm(alpha=ema_alpha, adjust=False).mean()
    final_alpha_factor = final_alpha_factor - ema_close
    
    return final_alpha_factor
