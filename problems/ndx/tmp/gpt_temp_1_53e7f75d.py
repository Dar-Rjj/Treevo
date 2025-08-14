def heuristics_v2(df):
    # Calculate the 20-day moving average of the close price
    df['20_day_ma'] = df['close'].rolling(window=20).mean()
    
    # Compute the percentage change from today's close to the 20-day moving average
    df['pct_change_20_day_ma'] = (df['close'] - df['20_day_ma']) / df['20_day_ma'] * 100
    
    # Smooth the percentage change with a 5-day EMA
    df['smoothed_pct_change_20_day_ma'] = df['pct_change_20_day_ma'].ewm(span=5).mean()
    
    # Calculate the 60-day volatility of the percentage change
    df['volatility_pct_change'] = df['pct_change_20_day_ma'].rolling(window=60).std()
    
    # Define the threshold for significant deviation as 2 standard deviations
    df['threshold_pct_change'] = 2 * df['volatility_pct_change']
    
    # Flag potential reversal if the smoothed percentage change exceeds the threshold
    df['reversal_signal'] = df['smoothed_pct_change_20_day_ma'].abs() > df['threshold_pct_change']
    
    # Calculate the 20-day weighted average volume, giving more weight to recent days
    df['20_day_avg_volume'] = df['volume'].ewm(span=20).mean()
    
    # Calculate the ratio of today's volume to the 20-day average volume
    df['volume_ratio'] = df['volume'] / df['20_day_avg_volume']
    
    # Introduce a combined score by multiplying the percentage deviation and the volume ratio
    df['combined_score'] = df['smoothed_pct_change_20_day_ma'] * df['volume_ratio']
    
    # Define a threshold for the combined score
    df['threshold_combined_score'] = 10  # This can be adjusted based on empirical analysis
    
    # Strong reversal signal if the combined score exceeds the threshold
    df['strong_reversal_signal'] = df['combined_score'].abs() > df['threshold_combined_score']
    
    # Calculate the slope of the 20-day moving average
    df['20_day_ma_slope'] = df['20_day_ma'].diff() / df['20_day_ma'].shift(1)
    
    # Enhance the reversal signal based on the slope
    df['enhanced_reversal_signal'] = df['strong_reversal_signal'] & (df['20_day_ma_slope'] * df['smoothed_pct_change_20_day_ma'] < 0)
    
    # Daily performance indicator
    df['daily_return'] = (df['close'] - df['open']) / df['open']
