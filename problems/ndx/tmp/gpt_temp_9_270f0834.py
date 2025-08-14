import pandas as pd

def heuristics_v2(df):
    # Calculate the 14-day RSI of the close price
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate the logarithmic change in volume
    log_volume_change = df['volume'].pct_change().apply(lambda x: 0 if pd.isna(x) else x).add(1).apply(np.log)
    
    # Generate the heuristic matrix by multiplying the RSI with the logarithmic change in volume
    heuristics_matrix = rsi * log_volume_change
    
    return heuristics_matrix
