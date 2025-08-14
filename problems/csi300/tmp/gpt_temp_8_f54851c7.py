import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Reversal
    df['Intraday_Reversal'] = 2 * (df['High'] - df['Low']) / (df['Close'] + df['Open'])
    
    # Adjust for Open Interest
    df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
    df['Adjusted_Intraday_Reversal'] = df['Intraday_Reversal'] * (1 + df['Volume_Change'])

    # Calculate Daily Price Momentum
    df['Daily_Price_Momentum'] = df['Close'] - df['Close'].shift(10)

    # Calculate Volume Surprise
    df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Surprise'] = df['Volume'] - df['Volume_MA10']

    # Calculate Intraday Range
    df['Intraday_Range'] = df['High'] - df['Low']

    # Calculate Intraday Midpoint
    df['Intraday_Midpoint'] = (df['High'] + df['Low']) / 2

    # Calculate Close to Midpoint Deviation
    df['Close_to_Midpoint_Deviation'] = df['Close'] - df['Intraday_Midpoint']

    # Consider Day-to-Day Open Price Change
    df['Day_to_Day_Open_Price_Change'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Combine Metrics
    df['Combined_Metrics'] = df['Intraday_Range'] + df['Close_to_Midpoint_Deviation'] + df['Day_to_Day_Open_Price_Change']

    # Compute Volume Influence Ratio
    df['Up_Volume'] = df.apply(lambda row: row['Volume'] if row['Close'] > row['Open'] else 0, axis=1)
    df['Down_Volume'] = df.apply(lambda row: row['Volume'] if row['Close'] < row['Open'] else 0, axis=1)
    df['Up_Volume_Sum'] = df['Up_Volume'].rolling(window=5).sum()
    df['Down_Volume_Sum'] = df['Down_Volume'].rolling(window=5).sum()
    df['Volume_Influence_Ratio'] = df['Up_Volume_Sum'] / df['Down_Volume_Sum']

    # Calculate Intraday High-Low Spread
    df['Intraday_High_Low_Spread'] = df['High'] - df['Low']

    # Calculate Intraday Close-Open Return
    df['Intraday_Close_Open_Return'] = (df['Close'] - df['Open']) / df['Open']

    # Calculate Volume Surge
    df['Volume_Surge'] = df['Volume'] - df['Volume'].shift(1)

    # Calculate 5-Day Moving Average of Intraday Close-Open Return
    df['MA5_Intraday_Close_Open_Return'] = df['Intraday_Close_Open_Return'].rolling(window=5).mean()

    # Calculate 5-Day Moving Average of Intraday High-Low Spread
    df['MA5_Intraday_High_Low_Spread'] = df['Intraday_High_Low_Spread'].rolling(window=5).mean()

    # Calculate Weighted Intraday Return
    df['Weighted_Intraday_Return'] = 0.5 * df['Intraday_Close_Open_Return'] + 0.5 * df['Intraday_High_Low_Spread']

    # Combine Weighted Intraday Return, Moving Averages, and Volume Surge
    df['Intraday_Momentum'] = (
        df['Weighted_Intraday_Return'] * df['MA5_Intraday_Close_Open_Return'] +
        df['Intraday_High_Low_Spread'] * df['MA5_Intraday_High_Low_Spread']
    ) * df['Volume_Surge']

    # Synthesize Final Alpha Factor
    df['Final_Alpha_Factor'] = (
        (df['Adjusted_Intraday_Reversal'] * df['Combined_Metrics']) +
        (df['Daily_Price_Momentum'] * df['Volume_Surprise'] * df['Intraday_Range'] * df['Volume_Influence_Ratio']) +
        df['Intraday_Momentum'] +
        df['Volume_Surge']
    )

    return df['Final_Alpha_Factor']
