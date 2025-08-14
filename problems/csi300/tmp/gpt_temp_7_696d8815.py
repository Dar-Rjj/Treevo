import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=10):
    # Calculate Simple Momentum
    df['Simple_Momentum'] = df['close'] - df['close'].shift(n)
    
    # Volume-Adjusted Component
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Adjusted_Momentum'] = df['Volume_Change'] * df['Simple_Momentum']
    
    # Enhanced Price Reversal Sensitivity
    df['High_Low_Spread'] = df['high'] - df['low']
    df['Open_Close_Spread'] = df['open'] - df['close']
    df['Weighted_High_Low_Spread'] = df['Volume'] * df['High_Low_Spread']
    df['Weighted_Open_Close_Spread'] = df['Volume'] * df['Open_Close_Spread']
    df['Combined_Weighted_Spreads'] = df['Weighted_High_Low_Spread'] + df['Weighted_Open_Close_Spread']
    
    # Smooth High-Low Spread
    df['Smoothed_High_Low_Spread'] = df['High_Low_Spread'].ewm(span=5).mean()
    
    # Combine Momentum and Close-to-Low Distance
    df['Close_to_Low_Distance'] = df['close'] - df['low']
    df['Momentum_Combined'] = df['Simple_Momentum'] * df['Close_to_Low_Distance']
    
    # Measure Volume Impact
    df['EMA_Volume_10'] = df['volume'].ewm(span=10).mean()
    
    # Compute Raw Momentum
    df['Raw_Momentum'] = df['close'] - df['close'].shift(n)
    
    # Adjust for Volume
    df['Average_Volume'] = df['volume'].rolling(window=n).mean()
    df['Volume_Ratio'] = df['volume'] / df['Average_Volume']
    df['Adjusted_Raw_Momentum'] = df['Raw_Momentum'] * df['Volume_Ratio']
    
    # Incorporate Enhanced Price Gaps and Volume Oscillations
    df['Open_to_Close_Gap'] = df['open'] - df['close']
    df['High_Low_Gap'] = df['high'] - df['low']
    df['Historical_Average_Volume'] = df['volume'].rolling(window=m).mean()
    df['Volume_Difference'] = df['volume'] - df['Historical_Average_Volume']
    df['Volume_Oscillation'] = df['Volume_Difference'] / df['Historical_Average_Volume']
    
    df['Weighted_Open_to_Close_Gap'] = 0.3 * df['Open_to_Close_Gap']
    df['Weighted_High_Low_Gap'] = 0.4 * df['High_Low_Gap']
    df['Weighted_Volume_Oscillation'] = 0.3 * df['Volume_Oscillation']
    
    df['Combined_Volume_Adjusted_Momentum'] = (df['Volume_Adjusted_Momentum'] 
                                               + df['Weighted_Open_to_Close_Gap'] 
                                               + df['Weighted_High_Low_Gap'] 
                                               + df['Weighted_Volume_Oscillation'])
    
    # Final Alpha Factor
    df['Final_Alpha_Factor'] = (df['Combined_Volume_Adjusted_Momentum'] 
                                - df['Combined_Weighted_Spreads'] 
                                + df['Adjusted_Raw_Momentum'] 
                                / df['EMA_Volume_10'])
    
    return df['Final_Alpha_Factor']
