import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    momentum = (close / close.shift(5) - 1) - (close / close.shift(20) - 1)
    volatility = (high / low).rolling(window=10).std()
    liquidity = (amount / volume).pct_change(periods=3)
    
    factor = momentum * np.log(1 + volatility) * np.tanh(liquidity)
    heuristics_matrix = pd.Series(factor, index=df.index, name='heuristics_v2')
    
    return heuristics_matrix
