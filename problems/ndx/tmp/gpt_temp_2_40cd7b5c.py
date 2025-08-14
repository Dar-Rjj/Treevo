import numpy as np
def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume-Weighted Price Change
    df['volume_weighted_price_change'] = df['daily_price_change'] * df['volume']
    
    # Sum Cumulative Volume-Weighted Price Changes
    df['cumulative_volume_weighted_price_change'] = df['volume_weighted_price_change'].cumsum()
    
    # Apply Exponential Moving Average to Cumulative Sum
    alpha = 0.1  # Define alpha based on lookback period
    df['ema_cumulative_sum'] = df['cumulative_volume_weighted_price_change'].ewm(alpha=alpha).mean()
    
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Incorporate Close Price Trend
    df['10_day_ma_close'] = df['close'].rolling(window=10).mean()
    df['trend'] = (df['close'] > df['10_day_ma_close']).astype(int)
    
    # Combine EMA, High-Low Range, and Close Price Trend
    df['intermediate_factor'] = np.where(
        df['trend'] == 1,
        0.7 * df['ema_cumulative_sum'] + 0.3 * df['high_low_range'],
        0.3 * df['ema_cumulative_sum'] + 0.7 * df['high_low_range']
    )
    
    # Adjust Momentum by Combined Volatility
    df['20_day_momentum'] = df['close'].pct_change(20)
    df['daily_volume_diff'] = df['volume'].diff().fillna(0)
    df['daily_amount_diff'] = df['amount'].diff().fillna(0)
    df['combined_volatility'] = df['daily_volume_diff'].rolling(window=22).std() + df['daily_amount_diff'].rolling(window=22).std()
    df['adjusted_momentum'] = df['20_day_momentum'] / df['combined_volatility']
    
    # Identify Breakout Days
    df['avg_high_low_range_21'] = df['high_low_range'].rolling(window=21).mean()
    df['is_breakout'] = (df['high_low_range'] > 2 * df['avg_high_low_range_21']).astype(int)
    
    # Calculate Volume-Adjusted Breakout Impact
    df['return_open_to_close'] = (df['close'] - df['open']) / df['open']
