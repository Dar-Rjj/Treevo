import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Price Momentum
    momentum_lookback = 20
    df['Price_Momentum'] = df['close'].pct_change(momentum_lookback)

    # Calculate Volume Acceleration
    volume_lookback = 10
    df['Volume_Acceleration'] = df['volume'].pct_change(volume_lookback)

    # Combine Momentum and Volume
    df['Intermediate_Factor'] = df['Price_Momentum'] * df['Volume_Acceleration']

    # Calculate High-to-Low Range
    df['High_Low_Range'] = df['high'] - df['low']

    # Adjust for Volume
    df['Adjusted_High_Low_Range'] = df['High_Low_Range'] * df['volume']

    # Detect Volume Spike
    df['Volume_7D_MA'] = df['volume'].rolling(window=7).mean()
    df['Volume_Spike_Flag'] = (df['volume'] > 1.8 * df['Volume_7D_MA']).astype(int)

    # Calculate Volume Trend
    volume_trend_window = 15
    df['Volume_Trend'] = (df['volume'] > df['volume'].rolling(window=volume_trend_window).mean()).astype(int)
    df['Volume_Trend'] = df['Volume_Trend'].replace({1: 1, 0: -1})

    # Calculate Intraday Return
    df['Intraday_Return'] = (df['high'] - df['low']) / df['low']

    # Intraday Volatility Adjusted
    df['Intraday_Volatility'] = (df[['high', 'low', 'close', 'open']].std(axis=1))
    df['Intraday_Volatility_Adjusted'] = df['Intraday_Return'] / df['Intraday_Volatility']

    # Calculate Volume Displacement
    df['Volume_Displacement'] = df['volume'] - df['volume'].shift(1)

    # Combine Intraday Factors
    df['Combined_Intraday_Factors'] = df['Intraday_Volatility_Adjusted'] * df['Volume_Displacement']

    # Calculate Trading Range Expansion
    trading_range_lookback = 20
    df['True_Range'] = df[['high', 'low']].apply(lambda x: max(x[0], df['close'].shift(1)) - min(x[1], df['close'].shift(1)), axis=1)
    df['Average_True_Range'] = df['True_Range'].rolling(window=trading_range_lookback).mean()

    # Final Combination
    df['Final_Factor'] = (df['Adjusted_High_Low_Range'] * df['Volume_Trend'] + 
                          df['Intermediate_Factor'] + 
                          2 * df['Intermediate_Factor'] * df['Volume_Spike_Flag'] + 
                          df['Combined_Intraday_Factors'] + 
                          df['Volume_Acceleration'] + 
                          df['Average_True_Range'])

    return df['Final_Factor']
