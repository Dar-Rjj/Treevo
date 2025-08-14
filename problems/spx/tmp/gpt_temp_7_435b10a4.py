def heuristics_v2(df):
    # Compute Exponential Moving Averages (EWMAs)
    df['5_day_ema_close'] = df['close'].ewm(span=5, adjust=False).mean()
    df['20_day_ema_close'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Determine Trend Direction
    df['trend_direction'] = df['5_day_ema_close'] - df['20_day_ema_close']
    
    # Volume Growth and Confirmation
    df['daily_volume_growth'] = df['volume'].pct_change()
    df['5_day_vol_sum'] = df['volume'].rolling(window=5).sum()
    
    # Identify Volume Spikes
    m = 20
    df['avg_volume'] = df['volume'].rolling(window=m).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['avg_volume']).astype(int)
    
    # Adjust Trend by Volume Spike
    scaling_factor = 1.5
    df['adjusted_trend'] = df['trend_direction'] * (1 + (scaling_factor - 1) * df['volume_spike'])
    
    # Evaluate Daily Trading Range
    df['daily_range'] = df['high'] - df['low']
    df['20_day_ema_range'] = df['daily_range'].ewm(span=20, adjust=False).mean()
    
    # Intraday Volatility Adjusted Price Change
    df['daily_price_change'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['intraday_vol_adj_price_change'] = df['daily_price_change'] / df['daily_range']
    
    # Combine Indicators
    df['combined_indicator'] = 0.5 * df['5_day_ema_close'] + 0.3 * df['5_day_vol_sum'] + 0.2 * df['20_day_ema_range']
    df['combined_indicator'] += df['intraday_vol_adj_price_change']
    
    # Smooth Using 5-day Simple Moving Average
    df['smoothed_combined_indicator'] = df['combined_indicator'].rolling(window=5).mean()
    
    # Weight by Trade Intensity
    df['vwap'] = (df['amount'] / df['volume'])
    df['trade_intensity'] = df['vwap'] / df['close']
    df['weighted_combined_indicator'] = df['smoothed_combined_indicator'] * df['trade_intensity']
    
    # Calculate High-to-Low Ratio
    df['high_to_low_ratio'] = df['high'] / df['low']
    
    # Compute High-to-Low Return
    df['high_to_low_return'] = df['high_to_low_ratio'] - 1
