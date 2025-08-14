import pandas as pd

def heuristics_v2(df):
    # Directional Movement Index (DMI)
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = -df['low'].diff()
    df['pos_dm'] = df.apply(lambda row: row['high_diff'] if (row['high_diff'] > row['low_diff']) and (row['high_diff'] > 0) else 0, axis=1)
    df['neg_dm'] = df.apply(lambda row: row['low_diff'] if (row['low_diff'] > row['high_diff']) and (row['low_diff'] > 0) else 0, axis=1)
    tr = df['high'] - df['low']
    atr = tr.rolling(window=14).mean()
    pos_di = (df['pos_dm'].ewm(span=14, adjust=False).mean() / atr) * 100
    neg_di = (df['neg_dm'].ewm(span=14, adjust=False).mean() / atr) * 100
    
    # Volatility Clustering
    log_returns = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    volatility = log_returns.rolling(window=20).std() * np.sqrt(252)
    
    # Liquidity Measure
    liquidity = df['volume'] / df['amount']
    
    # Composite Heuristic
    heuristics_matrix = (pos_di - neg_di + volatility + liquidity) / 4
    return heuristics_matrix
