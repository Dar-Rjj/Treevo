import pandas as pd

def heuristics_v2(df):
    heuristics_matrix = pd.Series(index=df.index)
    
    # Example heuristic: ratio of average volume in the last 5 days to the last 20 days
    heuristics_matrix['vol_ratio_5_20'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    
    # Price change over the last 5 days
    heuristics_matrix['price_change_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Relative strength index (RSI) calculation for the close prices
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    heuristics_matrix['rsi_close'] = rsi
    
    # Standard deviation of the last 30 day's closing prices as a measure of volatility
    heuristics_matrix['std_30d'] = df['close'].rolling(window=30).std()
    
    return heuristics_matrix
