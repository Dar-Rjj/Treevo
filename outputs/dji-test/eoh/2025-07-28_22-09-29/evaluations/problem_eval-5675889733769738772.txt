import pandas as pd

def heuristics_v2(df):
    # Calculate the 10-day exponential moving average of the absolute difference between close and open
    df['Trend_Signal'] = (df['close'] - df['open']).abs().ewm(span=10, adjust=False).mean()
    
    # Calculate the 10-day exponential moving average of the True Range
    df['True_Range'] = df[['high'-'low', 'high'-'close.shift(1)', 'close.shift(1)'-'low']].max(axis=1)
    df['Avg_True_Range'] = df['True_Range'].ewm(span=10, adjust=False).mean()
    
    # Trend following component
    df['Trend_Following'] = df['Trend_Signal'] / df['Avg_True_Range']
    
    # Rate of Change in Volume over 10 days
    df['Volume_RoC'] = df['volume'].pct_change(periods=10)
    
    # Generate the heuristics factor
    df['Heuristic_Factor'] = df['Trend_Following'] * df['Volume_RoC']
    
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    return heuristics_matrix
