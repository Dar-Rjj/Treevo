import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Daily Gain or Loss
    df['Daily_Gain_Loss'] = df['close'].diff()
    df['Sign_Daily_Gain_Loss'] = df['Daily_Gain_Loss'].apply(lambda x: 1 if x > 0 else -1)
    
    # Volume and Price Adjusted Gain/Loss
    df['Adjusted_Gain_Loss'] = df['Daily_Gain_Loss'] * df['volume'] * df['close']
    
    # Sum Past 5 Days' Adjusted Values
    df['Cumulated_Adjusted_Value'] = df['Adjusted_Gain_Loss'].rolling(window=5).sum()
    
    # Calculate Cumulative Volume
    df['Cumulative_Volume'] = df['volume'].rolling(window=5).sum()
    
    # Multiply Daily Returns by Cumulative Volume
    df['Daily_Returns'] = df['close'].pct_change()
    df['Return_Volume_Product'] = df['Daily_Returns'] * df['Cumulative_Volume']
    
    # Compute Cumulative Average
    df['Cumulative_Average'] = df['Return_Volume_Product'].rolling(window=5).mean()
    
    # Integrate Cumulative Average and Cumulated Adjusted Value
    df['Integrated_Cumulative'] = df['Cumulative_Average'] * df['Cumulated_Adjusted_Value']
    
    # Calculate Simple Moving Averages (SMA) of Close Prices
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    
    # Compute Momentum Difference
    df['Momentum_Difference'] = (df['SMA_20'] - df['SMA_5']).abs()
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['close'] * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # Compute Adjusted Momentum
    df['Adjusted_Momentum'] = df['Momentum_Difference'] * df['VWAP'] * df['Momentum_Difference'].apply(lambda x: 1 if x > 0 else -1)
    
    # Calculate Intraday Price Range Change
    df['Intraday_Price_Range_Change'] = (df['high'] - df['low']) - (df['high'].shift(1) - df['low'].shift(1))
    
    # Calculate Volume Change
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    
    # Combine Range and Volume Change
    df['Combined_Range_Volume_Change'] = df['Intraday_Price_Range_Change'] + df['Volume_Change']
    
    # Apply Sign Function for Directional Bias
    df['Directional_Bias'] = df['Combined_Range_Volume_Change'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
    
    # Construct Volatility Adjustment
    df['Realized_Volatility'] = df['close'].rolling(window=5).std()
    df['Volatility_Adjustment'] = 1 / (df['Realized_Volatility'] + 1e-8)
    
    # Combine Adjusted Momentum and Volatility
    df['Combined_Adjusted_Momentum_Volatility'] = df['Adjusted_Momentum'] * df['Volatility_Adjustment']
    
    # Integrate Combined Alpha Factor
    df['Integrated_Alpha_Factor'] = df['Combined_Adjusted_Momentum_Volatility'] * df['Directional_Bias']
    df['Integrated_Alpha_Factor'] = df.apply(
        lambda row: row['Integrated_Alpha_Factor'] if row['Directional_Bias'] != 0 else row['Integrated_Alpha_Factor'] * row['Adjusted_Momentum'].apply(lambda x: 1 if x > 0 else -1),
        axis=1
    )
    
    # Add Trend Following Component
    M = 10  # Define Moving Average Length M
    df['SMA_Trend'] = df['close'].rolling(window=M).mean()
    df['Trend_Signal'] = df.apply(lambda row: 1 if row['close'] > row['SMA_Trend'] else 0, axis=1)
    
    # Combine Final Alpha Factor with Trend Signal
    df['Final_Alpha_Factor'] = df['Integrated_Alpha_Factor'] * df['Trend_Signal']
    
    return df['Final_Alpha_Factor']
