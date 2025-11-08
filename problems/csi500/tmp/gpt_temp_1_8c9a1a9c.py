import pandas as pd

def heuristics_v2(df):
    # Relative strength breakout detection
    close_rank = df['close'].rolling(window=20).apply(lambda x: pd.Series(x).rank().iloc[-1])
    high_breakout = (df['close'] > df['high'].shift(1)) & (close_rank > 15)
    
    # Trend persistence with acceleration
    trend_strength = (df['close'] - df['close'].rolling(window=10).mean()) / df['close'].rolling(window=10).std()
    liquidity_accel = df['volume'].diff(3) / df['volume'].rolling(window=10).mean()
    accelerated_trend = trend_strength * liquidity_accel
    
    # Combine breakout signals with accelerated trends
    heuristics_matrix = high_breakout.astype(float) + accelerated_trend
    
    return heuristics_matrix
