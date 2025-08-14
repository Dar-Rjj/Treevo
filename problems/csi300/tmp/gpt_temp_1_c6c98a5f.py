import pandas as pd
import ta

def heuristics_v2(df):
    # Calculate the EMA of the closing prices with a 14-day window
    ema = ta.trend.ema_indicator(df['close'], window=14)
    
    # Calculate the Money Flow Index (MFI) over the last 14 days
    mfi = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
    
    # Calculate the Relative Strength Index (RSI) over the last 14 days
    rsi = ta.momentum.rsi(df['close'], window=14)
    
    # Combine the factors into a heuristics matrix, assigning equal weights for simplicity
    heuristics_matrix = (ema + mfi + rsi) / 3
    
    return heuristics_matrix
