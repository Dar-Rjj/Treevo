import pandas as pd

def heuristics_v2(df):
    # Logarithmic Return in Closing Price
    log_return = df['close'].apply(lambda x: np.log(x)) - df['close'].shift().apply(lambda x: np.log(x))
    
    # True Range (TR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate RSI of True Range (RSI-TR) over 14 days
    delta = true_range.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi_tr = 100 - (100 / (1 + rs))
    
    # Heuristics matrix combining log return and RSI-TR
    heuristics_matrix = log_return + rsi_tr
    return heuristics_matrix
