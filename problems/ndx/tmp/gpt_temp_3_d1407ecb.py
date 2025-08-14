def heuristics_v2(df):
    # Calculate the 5-day and 20-day moving averages of closing prices
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Momentum indicator: difference between short-term and long-term moving averages
    df['momentum_indicator'] = df['ma_5'] - df['ma_20']
    
    # 14-day Exponential Moving Average (EMA) of the closing price
    df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # Difference between current close and 14-day EMA
    df['price_ema_diff'] = df['close'] - df['ema_14']
    
    # On-Balance Volume (OBV) adjusted by the closing price
    df['obv'] = (df['close'] > df['close'].shift(1)).astype(int) * df['volume'] - (df['close'] < df['close'].shift(1)).astype(int) * df['volume']
    df['obv'] = df['obv'].cumsum()
    df['obv_adjusted'] = df['obv'] / df['close']
    
    # Ratio of volume of days with positive return over the total volume in the past 30 trading days
