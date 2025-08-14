import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Compute Intraday High-Low Spread
    df['Intraday_High_Low_Spread'] = df['High'] - df['Low']
    
    # Compute Previous Day's Close-to-Open Return
    df['Prev_Close_Open_Return'] = df['Close'].shift(1) - df['Open']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['Daily_VWAP'] = ((df['Open'] + df['High'] + df['Low'] + df['Close']) / 4) * df['Volume']
    total_volume = df['Volume'].sum()
    df['Daily_VWAP'] = df['Daily_VWAP'].cumsum() / df['Volume'].cumsum()
    
    # Combine Intraday Momentum and VWAP
    df['Combined_Value'] = df['Daily_VWAP'] - df['Intraday_High_Low_Spread']
    
    # Weight by Intraday Volume
    df['Weighted_Combined_Value'] = df['Combined_Value'] * df['Volume']
    
    # Smooth the Factor using Exponential Moving Average (EMA)
    df['Alpha_Factor'] = df['Weighted_Combined_Value'].ewm(span=5, adjust=False).mean()
    
    return df['Alpha_Fctor']

# Example usage:
# df = pd.DataFrame({
#     'Open': [...],
#     'High': [...],
#     'Low': [...],
#     'Close': [...],
#     'Amount': [...],
#     'Volume': [...]
# }, index=pd.DatetimeIndex([...]))
# alpha_factor = heuristics_v2(df)
