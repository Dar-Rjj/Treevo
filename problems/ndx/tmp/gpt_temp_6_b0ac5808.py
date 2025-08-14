import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Log Return
    df['Daily_Log_Return'] = (df['close'].shift(0) / df['close'].shift(1)).apply(lambda x: 0 if x == 0 else math.log(x))

    # Calculate Volume Change
    df['Volume_Change'] = df['volume'].shift(0) - df['volume'].shift(1)
    
    # Combine Momentum and Volume
    df['Preliminary_Factor'] = df['Daily_Log_Return'] * df['Volume_Change']

    # Apply Price Filter
    df['Avg_Price_t'] = (df['open'].shift(0) + df['close'].shift(0)) / 2
    df['Avg_Price_t-1'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    df['Preliminary_Factor_Adj'] = df.apply(lambda row: row['Preliminary_Factor'] * 1.1 if row['Avg_Price_t'] > row['Avg_Price_t-1'] else row['Preliminary_Factor'] * 0.9, axis=1)

    # Apply Volume Spike Filter
    df['Volume_Spike'] = (df['volume'].shift(0) > 2 * df['volume'].shift(1)).astype(int)
    df['Preliminary_Factor_Adj'] = df['Preliminary_Factor_Adj'] * (1 + df['Volume_Spike'])

    # Incorporate Intraday Dynamics
    df['Intraday_Momentum'] = df['high'] - df['low']
    df['Volume_Weighted_Intraday_Momentum'] = df['Intraday_Momentum'] * df['volume'].shift(0)

    # Identify Volume Shock
    df['Volume_Shock'] = (df['volume'].shift(0) > 1.5 * df['volume'].shift(1)).astype(int)

    # Smooth Realized Volatility
    df['Realized_Volatility'] = ((df['high'] - df['low']) / df['close']).rolling(window=20).std()
    df['Smoothed_Volatility'] = df['Realized_Volatility'].ewm(span=20, adjust=False).mean()

    # Adjust Combined Momentum Factor with Shock and Volatility
    df['Final_Factor'] = (df['Preliminary_Factor_Adj'] * df['Volume_Shock']) - df['Smoothed_Volatility']

    return df['Final_Factor']
