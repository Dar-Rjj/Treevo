def heuristics_v2(df):
    # Simple Moving Averages
    sma_14 = df['close'].rolling(window=14).mean()
    sma_28 = df['close'].rolling(window=28).mean()
    sma_ratio = (sma_14 / sma_28) - 1
    
    # Dynamic Lookback Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=21).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=21).mean()
    vol_lookback = (df['close'].rolling(window=21).std() * 21).fillna(21).astype(int)
    rsi = 100 - (100 / (1 + (gain / loss).rolling(window=vol_lookback).mean()))
    
    # Modified Money Flow Index (MFI) with 14-day window
    typical_price = (df['high'] + df['low'] + df['open'] + df['close']) / 4
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    money_flow_ratio = positive_flow.rolling(window=14).sum() / negative_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    # Volume weighted by the average of open, high, low, and close prices
    volume_weighted = df['volume'] * ((df['open'] + df['high'] + df['low'] + df['close']) / 4)
    
    # Composite heuristic
    heuristics_matrix = (sma_ratio + rsi + mfi + volume_weighted) / 4
    return heuristics_matrix
