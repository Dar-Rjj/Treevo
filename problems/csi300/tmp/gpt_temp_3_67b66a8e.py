import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread with Directional Bias
    df['HL_Spread'] = df['high'] - df['low']
    df['HL_Adjusted'] = df.apply(lambda row: row['HL_Spread'] * 1.5 if row['close'] > row['open'] else row['HL_Spread'] * 0.5, axis=1)
    
    # Compute Intraday Range
    df['Intraday_Range'] = df['high'] - df['low']
    
    # Adjust for Price Direction
    df['Intraday_Adjusted'] = df.apply(lambda row: row['Intraday_Range'] + (row['close'] - row['open']) if row['close'] > row['open'] else row['Intraday_Range'] - (row['close'] - row['open']), axis=1)
    
    # Calculate Daily Price Returns with Volume Spike Adjustment
    df['Daily_Return'] = df['close'].pct_change()
    df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Spike'] = df['volume'] > 2.5 * df['Volume_MA_20']
    df['Volume_Adjusted_Return'] = df.apply(lambda row: row['Daily_Return'] * 3.5 if row['Volume_Spike'] else row['Daily_Return'], axis=1)
    
    # Integrate Spread, Intraday Range, and Volume Adjusted Returns
    df['Integrated_Factor'] = df['HL_Adjusted'] * df['Intraday_Adjusted'] * df['Volume_Adjusted_Return']
    
    # Calculate Moving Averages
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    
    # Compute Crossover Signal
    df['Crossover_Signal'] = df['MA_10'] - df['MA_50']
    
    # Generate Alpha Factor
    df['Alpha_Factor'] = df.apply(lambda row: 1 if row['Crossover_Signal'] > 0 else -1, axis=1)
    
    # Integrate Combined Indicator and Alpha Factor
    df['Final_Alpha_Factor'] = df.apply(lambda row: row['Alpha_Factor'] + row['Integrated_Factor'] if row['Alpha_Factor'] == 1 else row['Alpha_Factor'] - row['Integrated_Factor'], axis=1)
    
    # Enhance Alpha Factor with Momentum Indicator
    df['Momentum_10'] = df['close'] - df['close'].shift(10)
    df['Enhanced_Alpha_Factor'] = df.apply(lambda row: row['Final_Alpha_Factor'] * 1.2 if row['Momentum_10'] > 0 else row['Final_Alpha_Factor'] * 0.8, axis=1)
    
    return df['Enhanced_Alpha_Factor']
