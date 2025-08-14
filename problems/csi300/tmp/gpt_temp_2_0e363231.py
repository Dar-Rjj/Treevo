import pandas as pd

def heuristics_v2(df):
    # Logarithmic Return in Closing Price
    log_return = df['close'].apply(np.log).diff()
    
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Exponential Moving Average of True Range (EMA)
    ema_true_range = true_range.ewm(span=20, adjust=False).mean()
    
    # Heuristics matrix combining log return and EMA of true range
    heuristics_matrix = log_return + ema_true_range
    return heuristics_matrix
