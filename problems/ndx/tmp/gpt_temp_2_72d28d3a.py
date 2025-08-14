import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 21-day Price Momentum
    df['price_momentum_21'] = df['close'] - df['close'].shift(21)
    
    # Identify Breakout Days
    avg_high_low_range_21 = df['high_low_range'].rolling(window=21).mean()
    df['breakout'] = (df['high'] - df['low']) > 3 * avg_high_low_range_21
    
    # Calculate Volume-Adjusted Breakout Impact
    df['volume_adjusted_breakout'] = df['breakout'] * (df['close'] - df['open']) * df['volume']
    df['sum_volume_adjusted_breakout'] = df['volume_adjusted_breakout'].rolling(window=21).sum()
    
    # Calculate Intraday Price Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close to Open Difference
    df['close_open_diff'] = df['close'] - df['open']
    
    # Calculate Volume-Weighted Price Change (Intraday)
    df['volume_weighted_intraday_change'] = df['close_open_diff'] * df['volume']
    
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Price Change (Daily)
    df['volume_weighted_daily_change'] = df['daily_price_change'] * df['volume']
    
    # Sum Cumulative Volume-Weighted Price Changes
    df['cumulative_volume_weighted_change'] = df['volume_weighted_daily_change'].rolling(window=21).sum()
    
    # Apply Exponential Moving Average to Cumulative Sum
    alpha = 0.2
    df['ema_cumulative_change'] = df['cumulative_volume_weighted_change'].ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate Volume Growth Rate
    df['volume_growth_rate'] = df['volume'] / df['volume'].shift(1)
    
    # Calculate Volume-Adjusted Momentum
    df['volume_adjusted_momentum'] = df['close'] - df['close'].shift(1) * df['volume']
    
    # Incorporate Close Price Trend
    df['7_day_ma_close'] = df['close'].rolling(window=7).mean()
    df['close_trend'] = (df['close'] > df['7_day_ma_close']).astype(int)
    
    # Combine High-Low Range, Volume-Adjusted Momentum, and Close Price Trend
    df['weighted_factors'] = (df['high_low_range'].rolling(window=21).sum() + 
                              df['volume_adjusted_momentum'].rolling(window=21).sum()) * 
                             (1 + df['close_trend'])
    
    # Integrate Volume Trend Impact
    df['volume_trend_factor'] = (df['volume'] - df['volume'].shift(1)).ewm(span=5, adjust=False).mean()
    
    # Adjust EMA Momentum by Combined Volatility
    df['combined_volatility'] = df['volume_trend_factor'].std() + (df['amount'].pct_change().abs().ewm(span=5, adjust=False).mean())
    df['adjusted_ema_momentum'] = df['ema_cumulative_change'] / df['combined_volatility'].replace(0, 1)
    
    # Combine All Factors
    df['combined_factors'] = (df['price_momentum_21'] + 
                              df['sum_volume_adjusted_breakout']) * 
                             df['volume_trend_factor'] * 
                             df['weighted_factors']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['combined_factors'] * df['volume_growth_rate']
    
    # Introduce Volume Spike Indicator
    df['volume_spike'] = (df['volume'] > 2.5 * df['volume'].rolling(window=30).mean()).astype(int)
    df['final_alpha_factor'] = df['final_alpha_factor'] * (1 + 0.7 * df['volume_spike'])
    
    return df['final_alpha_factor']
