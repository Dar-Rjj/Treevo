def heuristics_v2(df):
    # Calculate 20-day moving average of close prices
    df['20_day_ma'] = df['close'].rolling(window=20).mean()
    
    # Compute the percentage change from today's close to the 20-day moving average
    df['price_deviation'] = (df['close'] - df['20_day_ma']) / df['20_day_ma']
    
    # Flag significant deviations (e.g., more than 2 standard deviations) from historical percentage changes as potential reversal signals
    std_price_deviation = df['price_deviation'].rolling(window=20).std()
    df['reversal_signal'] = (df['price_deviation'].abs() > 2 * std_price_deviation).astype(int)
    
    # Calculate the ratio of today's volume to the 20-day average volume
    df['20_day_avg_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['20_day_avg_volume']
    
    # Introduce a condition that if the 20-day moving average is increasing, the reversal signal should be stronger if the close price is below the moving average, and vice versa
    df['ma_trend'] = df['20_day_ma'].diff().gt(0).astype(int)
    df['strong_reversal_signal'] = ((df['ma_trend'] == 1) & (df['close'] < df['20_day_ma'])) | ((df['ma_trend'] == 0) & (df['close'] > df['20_day_ma']))
    df['strong_reversal_signal'] = df['strong_reversal_signal'].astype(int)
    
    # Define a simple indicator as the difference between today's close and open, divided by the open
    df['daily_return'] = (df['close'] - df['open']) / df['open']
