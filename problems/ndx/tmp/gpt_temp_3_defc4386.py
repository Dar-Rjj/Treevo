import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate 20-Day Price Momentum
    df['20D_Price_Momentum'] = df['close'].pct_change(periods=20)
    
    # Compute 22-Day Combined Volatility
    df['Volume_Diff'] = df['volume'].diff()
    df['Amount_Diff'] = df['amount'].diff()
    df['22D_Volume_Std'] = df['Volume_Diff'].rolling(window=22).std()
    df['22D_Amount_Std'] = df['Amount_Diff'].rolling(window=22).std()
    df['22D_Combined_Volatility'] = df['22D_Volume_Std'] + df['22D_Amount_Std']
    
    # Adjust Momentum by Combined Volatility
    df['Adjusted_Momentum'] = df['20D_Price_Momentum'] / (df['22D_Combined_Volatility'] + 1e-6)
    
    # Calculate High-Low Range
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Calculate Open-to-Close Return
    df['Open_to_Close_Return'] = (df['close'] - df['open']) / df['open']
    
    # Volume-Weighted Average Return
    df['Volume_Weighted_Return'] = (df['return'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Daily Volume-Adjusted Momentum
    df['Daily_Volume_Adjusted_Momentum'] = (df['close'] - df['close'].shift(1)) * df['volume']
    
    # Incorporate Close Price Trend
    df['7D_MA_Close'] = df['close'].rolling(window=7).mean()
    df['Close_Price_Trend'] = df.apply(lambda row: 1 if row['close'] > row['7D_MA_Close'] else -1, axis=1)
    
    # Identify Breakout Days
    df['21D_Avg_High_Low_Range'] = df['High_Low_Range'].rolling(window=21).mean()
    df['Breakout_Day'] = (df['High_Low_Range'] > 2 * df['21D_Avg_High_Low_Range']).astype(int)
    
    # Calculate Volume-Adjusted Breakout Impact
    df['Volume_Adjusted_Breakout_Impact'] = (df['return'] * df['volume'] * df['Breakout_Day']).rolling(window=21).sum()
    
    # Integrate Volume Trend Impact
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['EMA_Volume_Trend'] = df['volume'].ewm(span=7).mean()
    
    # Combine High-Low Range, Volume-Adjusted Momentum, and Close Price Trend
    df['Combined_Factor'] = df['High_Low_Range'] * (1 + df['Close_Price_Trend']) + df['Daily_Volume_Adjusted_Momentum'] * (1 - df['Close_Price_Trend'])
    
    # Adjust by Open-to-Close Return
    df['Adjusted_Combined_Factor'] = df['Combined_Factor'] + df['Open_to_Close_Return'] * df['Open_to_Close_Return'].apply(lambda x: 1 if x > 0 else -1)
    
    # Incorporate Intraday Reversal
    df['Intraday_Reversal'] = (df['high'] - df['low']) / (df['high'] + df['low'])
    df['Volume_Adjusted_Reversal'] = df['Intraday_Reversal'] * df['volume']
    
    # Combine All Factors
    df['Final_Factor'] = df['Adjusted_Momentum'] + df['Volume_Adjusted_Breakout_Impact'] * df['EMA_Volume_Trend'] * df['Adjusted_Combined_Factor']
    
    # Final Alpha Factor
    df['Alpha_Factor'] = df['Final_Factor'] * (df['volume'] / df['volume'].shift(1))
    
    return df['Alpha_Factor']
