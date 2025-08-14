import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Close-Open Spread
    df['close_open_spread'] = df['close'] - df['open']
    
    # Calculate 14-day Price Momentum
    df['14_day_momentum'] = df['close'] - df['close'].shift(14)
    
    # Identify Breakout Days
    avg_high_low_range = df['high_low_range'].rolling(window=14).mean()
    df['breakout_day'] = (df['high_low_range'] > 2 * avg_high_low_range).astype(int)
    
    # Volume-Adjusted Breakout Impact
    df['daily_return'] = (df['close'] / df['open']) - 1
    df['volume_adjusted_return'] = df['daily_return'] * df['volume']
    df['volume_adjusted_breakout_impact'] = df['volume_adjusted_return'].rolling(window=14).sum() * df['breakout_day']
    
    # Intraday Momentum Indicator
    df['intraday_momentum_indicator'] = (df['close'] / df['open']) - 1
    
    # Volume-Weighted Close
    df['volume_weighted_close'] = df['volume'] * df['close']
    
    # Volume-Weighted Average Price
    df['volume_weighted_avg_price'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3) / df['volume']
    
    # Volume-Weighted Momentum
    df['volume_weighted_momentum'] = (df['volume_weighted_close'] / df['close']) - 1
    
    # Composite Intraday Indicator
    df['composite_intraday_indicator'] = df['intraday_momentum_indicator'] * df['volume_weighted_avg_price'] * df['volume_weighted_momentum']
    
    # Combine High-Low Range, Volume-Adjusted Momentum, and Close Price Trend
    trend = df['14_day_momentum'] > 0
    df['weighted_high_low_range'] = df['high_low_range'].rolling(window=14).sum()
    df['weighted_volume_adjusted_momentum'] = df['volume_adjusted_momentum'].rolling(window=14).sum()
    df['combined_factor'] = (df['weighted_high_low_range'] * ~trend) + (df['weighted_volume_adjusted_momentum'] * trend)
    
    # Integrate Volume Trend Impact
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_trend_ema'] = df['volume_change'].ewm(span=5).mean()
    
    # Combine All Factors
    df['combined_factors'] = (df['14_day_momentum'] + df['volume_adjusted_breakout_impact']) * df['volume_trend_ema'] * df['combined_factor']
    
    # Final Alpha Factor
    df['volume_growth_rate'] = df['volume'] / df['volume'].shift(1)
    df['final_alpha_factor'] = df['combined_factors'] * df['volume_growth_rate']
    
    # Introduce Volume Spike Indicator
    df['average_volume_20_days'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 2 * df['average_volume_20_days']).astype(float)
    
    # Adjust Final Alpha Factor by Volume Spike
    df['final_alpha_factor'] = df['final_alpha_factor'] * (1.5 * df['volume_spike'] + 1 * (1 - df['volume_spike']))
    
    return df['final_alpha_factor']
