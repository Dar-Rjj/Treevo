import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Price Movement Ratio
    df['High_Low_Ratio'] = df['high'] / df['low']
    
    # Evaluate Weighted Average Volume Over Period
    df['Average_Volume'] = df['volume'].rolling(window=20).mean()
    
    # Calculate Base Price Momentum
    df['Base_Momentum'] = df['close'] - df['close'].shift(20)
    
    # Adjust for Volume Impact
    df['Volume_Adjusted_Momentum'] = df.apply(
        lambda row: row['Base_Momentum'] * 1.1 if row['volume'] > row['Average_Volume'] else row['Base_Momentum'],
        axis=1
    )
    
    # Calculate Volume-Adjusted Momentum
    df['Volume_Adjusted_Momentum'] = (df['close'] - df['close'].shift(20)) / df['Average_Volume']
    
    # Assess Positive vs Negative Momentum Contribution
    df['Momentum_Contribution'] = df.apply(
        lambda row: row['Volume_Adjusted_Momentum'] * 1.5 if (row['close'] - row['close'].shift(20)) > 0 else row['Volume_Adjusted_Momentum'],
        axis=1
    )
    
    # Integrate Accumulated Momentum Impact
    # Sum of Adjusted Momentum from Multiple Time Horizons
    # Assign greater weight to longer time horizons
    df['Accumulated_Momentum_20'] = df['Momentum_Contribution'].rolling(window=20).sum() * 0.5
    df['Accumulated_Momentum_40'] = df['Momentum_Contribution'].rolling(window=40).sum() * 1.0
    df['Accumulated_Momentum_60'] = df['Momentum_Contribution'].rolling(window=60).sum() * 1.5
    
    df['Final_Factor'] = df['Accumulated_Momentum_20'] + df['Accumulated_Momentum_40'] + df['Accumulated_Momentum_60']
    
    return df['Final_Factor']
