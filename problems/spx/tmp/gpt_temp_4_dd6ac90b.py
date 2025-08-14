import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Move
    df['Intraday_Move'] = df['High'] - df['Close']
    
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = df['High'] - df['Low']
    
    # Calculate Daily Momentum
    df['Daily_Momentum'] = df['Close'] - df['Close'].shift(1)
    
    # Adjust Daily Momentum by Intraday Volatility
    df['Adjusted_Daily_Momentum'] = df['Daily_Momentum'] / df['Intraday_Volatility']
    
    # Estimate Trade Intensity
    df['Average_Price'] = (df['High'] + df['Low']) / 2
    df['Trade_Intensity'] = df['Volume'] / df['Average_Price']
    
    # Weight Intraday Move by Trade Intensity
    df['Weighted_Intraday_Move'] = df['Intraday_Move'] * df['Trade_Intensity']
    
    # Weight Adjusted Daily Momentum by Trade Intensity
    df['Weighted_Adjusted_Daily_Momentum'] = df['Adjusted_Daily_Momentum'] * df['Trade_Intensity']
    
    # Calculate Intraday Reversal
    df['Intraday_Reversal'] = df['High'] - df['Close']
    
    # Weight Intraday Reversal by Trade Intensity
    df['Weighted_Intraday_Reversal'] = df['Intraday_Reversal'] * df['Trade_Intensity']
    
    # Calculate Daily Volatility
    df['Daily_Volatility'] = df['High'] - df['Low']
    
    # Calculate Volume Change Ratio
    df['Volume_20d_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Change_Ratio'] = df['Volume'] / df['Volume_20d_MA']
    
    # Calculate Weighted Average Price
    df['Weighted_Average_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['Weighted_Price_Volume'] = df['Weighted_Average_Price'] * df['Volume_Change_Ratio']
    
    # Calculate Price Momentum
    df['Price_Momentum'] = df['Close'].pct_change(periods=5)
    
    # Identify Volume Trends
    df['Volume_Trend'] = df['Volume'] > df['Volume'].rolling(window=20).mean()
    
    # Adjust Momentum by Volume Trend
    df['Adjusted_Momentum'] = df['Price_Momentum'] + df['Volume_Trend'].astype(int) * 0.01
    
    # Smooth with 5-day Exponential Moving Average
    df['Smoothed_Momentum'] = df['Adjusted_Momentum'].ewm(span=5, adjust=False).mean()
    
    # Combine All Weighted Components
    df['Combined_Factor'] = (
        df['Weighted_Intraday_Move'] +
        df['Weighted_Adjusted_Daily_Momentum'] +
        df['Weighted_Intraday_Reversal']
    )
    
    # Calculate Final Factor
    df['Final_Factor'] = df['Combined_Factor'] * df['Daily_Volatility']
    
    return df['Final_Factor'].dropna()
