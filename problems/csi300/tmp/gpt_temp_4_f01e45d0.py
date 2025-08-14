import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Simple Moving Average (SMA) of the close price over a 10-day period
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    
    # Calculate Rate of Change (ROC) of the close price over short-term (5 days) and long-term (50 days) periods
    df['ROC_5'] = df['close'].pct_change(periods=5)
    df['ROC_50'] = df['close'].pct_change(periods=50)
    
    # Generate a signal for directional movement
    df['Direction_Signal'] = df['close'].diff().apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Compute the Volume-Weighted Average Price (VWAP)
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Identify days with significant price movements and high volumes
    df['Significant_Movement'] = (df['close'] - df['open']).abs() * df['volume']
    
    # Create a factor that represents the cumulative volume up to the current day
    df['Cumulative_Volume'] = df['volume'].cumsum()
    
    # Investigate the gap between the open and close prices
    df['Gap'] = df['open'] - df['close'].shift(1)
    df['Positive_Gap'] = df['Gap'].apply(lambda x: 1 if x > 0 else 0)
    df['Negative_Gap'] = df['Gap'].apply(lambda x: 1 if x < 0 else 0)
    df['Gap_Ratio'] = df['Positive_Gap'].rolling(window=10).sum() / (df['Negative_Gap'].rolling(window=10).sum() + 1e-6)
    
    # Examine the consistency of the direction of gaps
    df['Consecutive_Gaps'] = (df['Gap'] * df['Gap'].shift(1) > 0).astype(int)
    
    # Calculate the true range (TR)
    df['TR'] = df[['high', 'low']].diff(axis=1).iloc[:, 1].abs()
    df['TR'] = df.apply(lambda x: max(x['TR'], x['high'] - x['close'].shift(1), x['close'].shift(1) - x['low']), axis=1)
    
    # Compute the average true range (ATR) over a 14-day period
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    
    # Analyze the ratio of the high to the low price
    df['High_Low_Ratio'] = df['high'] / df['low']
    
    # Determine the difference between the high and the low price as a measure of intra-day volatility
    df['Intraday_Volatility'] = df['high'] - df['low']
    
    # Combine multiple factors into a single alpha factor
    df['Alpha_Factor'] = (
        df['SMA_10'].pct_change() + 
        df['ROC_5'] + 
        df['ROC_50'] + 
        df['Direction_Signal'] + 
        df['VWAP'].pct_change() + 
        df['Significant_Movement'].pct_change() + 
        df['Cumulative_Volume'].pct_change() + 
        df['Gap_Ratio'] + 
        df['Consecutive_Gaps'] + 
        df['ATR_14'].pct_change() + 
        df['High_Low_Ratio'].pct_change() + 
        df['Intraday_Volatility'].pct_change()
    )
    
    return df['Alpha_Factor']
