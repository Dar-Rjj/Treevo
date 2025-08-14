import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Momentum Component
    short_term_return = df['close'].pct_change(5)
    medium_term_return = df['close'].pct_change(10)
    long_term_return = df['close'].pct_change(20)
    combined_returns = 0.4 * short_term_return + 0.3 * medium_term_return + 0.3 * long_term_return
    
    # Calculate Intraday Movements
    intraday_high_low_spread = df['high'] - df['low']
    intraday_close_open_change = df['close'] - df['open']
    combined_intraday_movement = intraday_high_low_spread + intraday_close_open_change
    volume_amount_ratio = df['volume'] / df['amount']
    weighted_intraday_movement = combined_intraday_movement * volume_amount_ratio
    
    # Evaluate Recent Trend
    recent_trend = weighted_intraday_movement.rolling(window=5).mean()
    trend_comparison = weighted_intraday_movement - recent_trend
    
    # Introduce Reversal Indicator
    twenty_day_high = df['high'].rolling(window=20).max()
    twenty_day_low = df['low'].rolling(window=20).min()
    price_position_in_range = (df['close'] - twenty_day_low) / (twenty_day_high - twenty_day_low)
    volume_weighted_reversal = price_position_in_range * df['volume']
    
    # Combine Momentum, Intraday, and Volume-Weighted Reversal
    momentum_component = combined_returns
    intraday_component = trend_comparison
    reversal_component = volume_weighted_reversal
    combined_factor = momentum_component + intraday_component - reversal_component
    
    # Final Adjustment
    final_factor = combined_factor.apply(lambda x: x ** 0.5)
    stability_component = df['close'].rolling(window=10).mean()
    final_factor_adjusted = final_factor - stability_component
    
    return final_factor_adjusted
