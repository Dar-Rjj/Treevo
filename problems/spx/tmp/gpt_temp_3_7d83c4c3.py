import pandas as pd
import pandas as pd

def heuristics_v2(df, n=10):
    # Calculate Daily Price Momentum
    df['Close_diff'] = df['close'].diff(periods=n)
    df['Smoothed_Momentum'] = df['Close_diff'].rolling(window=n).sum()

    # Incorporate Volume Adjusted Inertia
    df['Positive_Volume'] = df.apply(lambda row: row['volume'] if row['close'] > df.loc[row.name - pd.Timedelta(days=1), 'close'] else 0, axis=1)
    df['Cumulative_Volume_Flow'] = df['Positive_Volume'].rolling(window=n).sum()
    
    # Combine Momentum with Cumulative Volume Flow
    df['Intermediate_Alpha_Factor'] = df['Smoothed_Momentum'] * df['Cumulative_Volume_Flow']

    # Calculate Intraday Return
    df['Intraday_Return'] = (df['close'] - df['open']) / df['open']

    # Calculate Volume Trend Factor
    df['Volume_5D_MA'] = df['volume'].rolling(window=5).mean()
    df['Volume_Trend'] = df['volume'] - df['Volume_5D_MA']
    
    # Multiply Intraday Return by Volume Trend
    df['Intraday_Momentum_with_Volume_Trend'] = df['Intraday_Return'] * df['Volume_Trend']

    # Synthesize Factors
    df['Final_Alpha_Factor'] = df['Intermediate_Alpha_Factor'] + df['Intraday_Momentum_with_Volume_Trend']

    return df['Final_Alpha_Factor']
