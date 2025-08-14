import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df):
    # Calculate Price Momentum
    df['Price_Momentum'] = df['close'].shift(10) - df['close']
    
    # Adjust for Volume Trend
    df['Volume_MA_10'] = df['volume'].rolling(window=10).mean()
    df['Volume_Adjustment'] = df['volume'] - df['Volume_MA_10']
    
    # Adjust for Amount Trend
    df['Amount_MA_10'] = df['amount'].rolling(window=10).mean()
    df['Amount_Adjustment'] = df['amount'] - df['Amount_MA_10']
    
    # Combine Price Momentum, Volume Adjustment, and Amount Adjustment
    df['Combined_Adjustment'] = df['Price_Momentum'] * df['Volume_Adjustment'] * df['Amount_Adjustment']
    df['Combined_Adjustment'] = df['Combined_Adjustment'].apply(lambda x: max(x, 0))
    
    # Calculate Daily Log Return
    df['Daily_Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Identify Positive and Negative Returns
    positive_returns = df[df['Daily_Log_Return'] > 0]
    negative_returns = df[df['Daily_Log_Return'] <= 0]
    
    # Calculate Sum of Upward Volume
    df['Upward_Volume'] = np.where(df['Daily_Log_Return'] > 0, df['volume'], 0)
    sum_upward_volume = df['Upward_Volume'].sum()
    
    # Calculate Sum of Downward Volume
    df['Downward_Volume'] = np.where(df['Daily_Log_Return'] <= 0, df['volume'], 0)
    sum_downward_volume = df['Downward_Volume'].sum()
    
    # Compute Trend Reversal Signal
    total_volume = sum_upward_volume + sum_downward_volume
    upward_volume_ratio = sum_upward_volume / total_volume
    downward_volume_ratio = sum_downward_volume / total_volume
    df['Trend_Reversal_Signal'] = np.where(upward_volume_ratio > downward_volume_ratio, 1, 0) * df['Daily_Log_Return']
    
    # Calculate Daily Price Change
    df['Daily_Price_Change'] = df['close'] - df['close'].shift(1)
    
    # Evaluate Momentum Over Time
    df['Sum_Daily_Price_Changes'] = df['Daily_Price_Change'].rolling(window=10).sum()
    
    # Compute Volume Weighted Momentum
    df['Average_Volume_10'] = df['volume'].rolling(window=10).mean()
    df['Volume_Weighted_Momentum'] = (df['Sum_Daily_Price_Changes'] * df['volume']) / df['Average_Volume_10']
    
    # Integrate Volume Spike Indicator
    df['Volume_Spike'] = np.where(df['volume'] > 1.5 * df['Average_Volume_10'], 1.2, 1)
    df['Volume_Weighted_Momentum'] = df['Volume_Weighted_Momentum'] * df['Volume_Spike']
    
    # Combine Price Momentum and Trend Reversal Signal
    df['Combined_Factor'] = df['Combined_Adjustment'] + df['Trend_Reversal_Signal']
    df['Combined_Factor'] = df['Combined_Fctor'].apply(lambda x: max(x, 0))
    
    # Final Alpha Factor
    df['Final_Alpha_Factor'] = df['Combined_Factor'] * df['Trend_Reversal_Signal'] + df['Volume_Weighted_Momentum']
    df['Final_Alpha_Factor'] = df['Final_Alpha_Factor'].apply(lambda x: max(x, 0))
    
    return df['Final_Alpha_Factor']
