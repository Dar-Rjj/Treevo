import pandas as pd

def heuristics_v2(df):
    # Weighted Moving Average
    wma_10 = df['close'].rolling(window=10).apply(lambda x: (x * range(1, 11)).sum() / sum(range(1, 11)))
    
    # Custom Volatility-Adjusted Momentum
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=20).std()
    momentum = (df['close'] / df['close'].shift(10) - 1) / volatility
    
    # Chaikin Money Flow
    m_flow = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']) * df['volume']
    cmf = m_flow.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Composite heuristic
    heuristics_matrix = (wma_10 + momentum + cmf) / 3
    return heuristics_matrix
