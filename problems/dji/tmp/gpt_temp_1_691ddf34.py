import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Price Range Percentage
    df['Price_Range_Percentage'] = (df['High'] - df['Low']) / df['Low'] * 100
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['Typical_Price'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
    df['Volume_Adjusted_TP'] = df['Typical_Price'] * df['Volume']
    vwap = df['Volume_Adjusted_TP'].sum() / df['Volume'].sum()
    df['VWAP'] = vwap
    
    # Analyze Volume Direction
    df['Volume_Flow'] = df.apply(lambda row: row['Volume'] if row['Close'] > row['Open'] else -row['Volume'], axis=1)
    
    # Calculate Intraday Volatility
    df['Intraday_Volatility'] = (df['High'] - df['Low']) / df['Low']
    
    # Calculate Money Flow Index (MFI)
    df['Typical_Price_MFI'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price_MFI'] * df['Volume']
    positive_money_flow = df[df['Close'] > df['Open']]['Raw_Money_Flow'].rolling(window=14).sum()
    negative_money_flow = df[df['Close'] <= df['Open']]['Raw_Money_Flow'].rolling(window=14).sum()
    money_flow_ratio = positive_money_flow / negative_money_flow
    df['MFI'] = 100 - (100 / (1 + money_flow_ratio))
    
    # Analyze Intraday Momentum
    df['Momentum'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Momentum_Category'] = df['Momentum'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    
    # Calculate Intraday Accumulation/Distribution Line
    df['Money_Flow_Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['Money_Flow_Volume'] = df['Money_Flow_Multiplier'] * df['Volume']
    df['Accumulation_Distribution'] = df['Money_Flow_Volume'].cumsum()
    
    # Return the factor values as a Series
    return df['Accumulation_Distribution']

# Example usage:
# df = pd.DataFrame({
#     'Date': ['2023-10-01', '2023-10-02', ...],
#     'Open': [100, 102, ...],
#     'High': [105, 107, ...],
#     'Low': [98, 101, ...],
#     'Close': [104, 106, ...],
#     'Volume': [1000, 1200, ...]
# })
# df.set_index('Date', inplace=True)
# factors = heuristics_v2(df)
