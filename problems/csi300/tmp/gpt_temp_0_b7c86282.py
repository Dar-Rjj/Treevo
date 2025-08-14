import pandas as pd

def heuristics_v2(df):
    # Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Mean Reversion Component
    mean_reversion = df['close'] - df['close'].rolling(window=20).mean()
    
    # Liquidity Indicator based on Logarithm of Cumulative Volume
    liquidity = (pd.Series.rolling((df['volume']).cumsum(), window=10).apply(lambda x: x[-1] if len(x) > 0 else 0)).apply(lambda x: 0 if x == 0 else math.log(x))
    
    # Combine the factors into a heuristics matrix, assigning equal weights for simplicity
    heuristics_matrix = (rsi + mean_reversion + liquidity) / 3
    
    return heuristics_matrix
