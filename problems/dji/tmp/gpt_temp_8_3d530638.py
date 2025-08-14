def heuristics_v2(df):
    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Volume change
    df['volume_change'] = df['volume'].pct_change()
    
    # Price momentum: 5-day moving average return
    df['momentum_5d'] = df['close'].pct_change(5)
    
    # Volume relative to 30-day average
    df['vol_over_avg_vol'] = df['volume'] / df['volume'].rolling(window=30).mean()
    
    # ATR (Average True Range) for volatility
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()

    # Calculate RSI over 10 days
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=10).mean()
    avg_loss = loss.rolling(window=10).mean()
    rs = avg_gain / avg_loss
    rsi_10 = 100 - (100 / (1 + rs))
    
    # 20-day price-to-moving-average ratio
    df['price_to_sma_20'] = df['close'] / df['close'].rolling(window=20).mean()
    
    # Compile heuristics into a matrix
    heuristics_matrix = pd.Series(index=df.index, dtype='float64')
    heuristics_matrix = df['daily_return'] + df['volume_change'] + df['momentum_5d'] + df['vol_over_avg_vol'] + df['atr_14'] + rsi_10 + df['price_to_sma_20']
    
    return heuristics_matrix
