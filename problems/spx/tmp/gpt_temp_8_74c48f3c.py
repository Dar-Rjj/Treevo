import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, m=10):
    # Calculate Intraday Move
    df['Intraday_Move'] = df['High'] - df['Close']
    
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = df['High'] - df['Low']
    
    # Calculate Adjusted Momentum
    df['Adjusted_Momentum'] = (df['Close'] / df['Close'].shift(n) - 1) / df['Intraday_Volatility']
    
    # Identify Volume Spikes
    df['Avg_Volume'] = df['Volume'].rolling(window=m).mean()
    df['Volume_Spike'] = df['Volume'] > (df['Avg_Volume'] * 1.5)
    scaling_factor = 2.0
    df['Adjusted_Momentum'] = df.apply(lambda row: row['Adjusted_Momentum'] * scaling_factor if row['Volume_Spike'] else row['Adjusted_Momentum'], axis=1)
    
    # Weight by Trade Intensity using VWAP
    df['VWAP'] = (df['Amount'] / df['Volume'])
    df['Trade_Intensity_VWAP'] = df['VWAP'] / ((df['High'] + df['Low']) / 2)
    df['Weighted_Adjusted_Momentum'] = df['Adjusted_Momentum'] * df['Trade_Intensity_VWAP']
    
    # Adjust Daily Momentum by Intraday Volatility
    df['Daily_Momentum'] = (df['Close'] / df['Close'].shift(1) - 1)
    df['Adjusted_Daily_Momentum'] = df['Daily_Momentum'] / df['Intraday_Volatility']
    
    # Weight Intraday Move by Trade Intensity
    df['Trade_Intensity_Intraday'] = df['Volume'] / ((df['High'] + df['Low']) / 2)
    df['Weighted_Intraday_Move'] = df['Intraday_Move'] * df['Trade_Intensity_Intraday']
    
    # Weight Adjusted Daily Momentum by Trade Intensity
    df['Weighted_Adjusted_Daily_Momentum'] = df['Adjusted_Daily_Momentum'] * df['Trade_Intensity_Intraday']
    
    # Calculate Intraday Reversal
    df['Intraday_Reversal'] = df['High'] - df['Close']
    
    # Weight Intraday Reversal by Trade Intensity
    df['Weighted_Intraday_Reversal'] = df['Intraday_Reversal'] * df['Trade_Intensity_Intraday']
    
    # Incorporate Intraday Range Percentage
    df['Intraday_Range_Percentage'] = (df['High'] - df['Low']) / df['Open']
    df['Adjusted_Intraday_Move'] = df['Intraday_Move'] * df['Intraday_Range_Percentage']
    
    # Weight Adjusted Intraday Move by Trade Intensity
    df['Weighted_Adjusted_Intraday_Move'] = df['Adjusted_Intraday_Move'] * df['Trade_Intensity_Intraday']
    
    # Combine All Weighted Components
    df['Alpha_Factor'] = (df['Weighted_Intraday_Move'] + 
                          df['Weighted_Adjusted_Momentum'] + 
                          df['Weighted_Adjusted_Daily_Momentum'] + 
                          df['Weighted_Intraday_Reversal'] + 
                          df['Weighted_Adjusted_Intraday_Move'])
    
    return df['Alpha_Factor']
