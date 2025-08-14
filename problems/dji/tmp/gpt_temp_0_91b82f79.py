import pandas as pd

def heuristics_v2(df):
    # Calculate the 10-day exponential moving average of log returns
    log_returns = df['close'].apply(lambda x: np.log(x)).diff()
    ema_log_returns = log_returns.ewm(span=10, adjust=False).mean()
    
    # Calculate the sum of the absolute differences between consecutive closing prices over the last 20 days
    abs_diff_sum = df['close'].diff().abs().rolling(window=20).sum()
    
    # Calculate the 14-day Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Combine factors into a single heuristics score with weights
    heuristics_matrix = (0.3 * ema_log_returns + 0.3 * abs_diff_sum + 0.4 * rsi)
    
    return heuristics_matrix
