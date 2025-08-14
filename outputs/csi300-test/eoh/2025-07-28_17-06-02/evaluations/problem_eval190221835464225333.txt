import pandas as pd
import ta

def heuristics_v2(df):
    # Calculate MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd_diff'] = macd.macd_diff()
    
    # Calculate RSI
    rsi = ta.momentum.RSIIndicator(close=df['close']).rsi()
    df['rsi'] = rsi
    
    # Create a composite heuristic factor
    df['heuristic_factor'] = df['macd_diff'] + df['rsi']
    
    # Extract the factor values as a pandas Series
    heuristics_matrix = df['heuristic_factor']
    
    return heuristics_matrix
