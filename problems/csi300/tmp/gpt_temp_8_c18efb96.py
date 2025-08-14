import pandas as pd

def heuristics_v2(df):
    # Weighted Moving Averages
    wma_7 = df['close'].rolling(window=7).apply(lambda x: (x * range(1, 8)).sum() / sum(range(1, 8)))
    wma_14 = df['close'].rolling(window=14).apply(lambda x: (x * range(1, 15)).sum() / sum(range(1, 15)))
    wma_ratio = (wma_7 / wma_14) - 1
    
    # Adaptive Relative Strength Index (RSI) with volatility adjustment
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    volatility = df['close'].rolling(window=14).std()
    rsi_adjusted = rsi / volatility
    
    # Modified Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df['volume']
    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
    money_flow_ratio = positive_flow.rolling(window=14).sum() / negative_flow.rolling(window=14).sum()
    mfi = 100 - (100 / (1 + money_flow_ratio))
    
    # Volume-adjusted Momentum
    momentum = df['close'] - df['close'].shift(14)
    volume_adjusted_momentum = momentum * (df['volume'] / df['volume'].rolling(window=14).mean())
    
    # Composite heuristic
    heuristics_matrix = (wma_ratio + rsi_adjusted + mfi + volume_adjusted_momentum) / 4
    return heuristics_matrix
