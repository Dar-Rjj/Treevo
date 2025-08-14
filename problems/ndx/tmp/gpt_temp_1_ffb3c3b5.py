import pandas as pd

def heuristics_v2(df):
    # Calculate the daily change in close price
    df['close_change'] = df['close'].diff()
    # Calculate the average true range (ATR) as a volatility measure
    df['tr'] = df[['high' - 'low', 'high' - df['close'].shift(1), 'low' - df['close'].shift(1)]].max(axis=1)
    atr = df['tr'].rolling(window=14).mean()
    # Scale the ATR by the volume to emphasize market activity
    scaled_atr = atr * df['volume']
    # Generate the heuristic factor by combining the close change with the scaled ATR
    heuristics_matrix = df['close_change'] + scaled_atr
    return heuristics_matrix
