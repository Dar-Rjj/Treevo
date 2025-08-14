import pandas as pd

def heuristics_v2(df):
    # Weighted Moving Averages
    weights_5 = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2])
    wma_5 = df['close'].rolling(window=5).apply(lambda x: (x * weights_5).sum(), raw=False)
    weights_10 = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    wma_10 = df['close'].rolling(window=10).apply(lambda x: (x * weights_10).sum(), raw=False)
    wma_ratio = (wma_5 / wma_10) - 1
    
    # Smoothed Relative Strength Index (RSI) with 14-day lookback
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Chaikin Money Flow (CMF) with 20-day window
    money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    money_flow_volume = money_flow_multiplier * df['volume']
    cmf = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Composite heuristic
    heuristics_matrix = (wma_ratio + rsi + cmf + typical_price) / 4
    return heuristics_matrix
