import pandas as pd

def heuristics_v2(df):
    heuristics_matrix = pd.DataFrame(index=df.index)
    
    # Example Alpha 1: Rate of change (close price)
    heuristics_matrix['Alpha1'] = df['close'].pct_change()
    
    # Example Alpha 2: Moving average convergence divergence (MACD) approximation
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    heuristics_matrix['Alpha2'] = short_ema - long_ema
    
    # Example Alpha 3: Price momentum over 5 periods
    heuristics_matrix['Alpha3'] = df['close'] / df['close'].shift(5) - 1
    
    # Example Alpha 4: Volume times rate of change of close price
    heuristics_matrix['Alpha4'] = df['volume'] * heuristics_matrix['Alpha1']
    
    return heuristics_matrix
