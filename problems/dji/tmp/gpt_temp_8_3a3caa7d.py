import pandas as pd

def heuristics_v2(df):
    momentum = df['close'].pct_change(periods=12)
    volatility = df['close'].pct_change().rolling(window=20).std()
    liquidity = df['volume'].pct_change().rolling(window=5).mean()
    heuristics_matrix = pd.concat([momentum, volatility, liquidity], axis=1)
    heuristics_matrix.columns = ['Momentum', 'Volatility', 'Liquidity']
    return heuristics_matrix
