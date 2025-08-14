import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['High-Low_Spread'] = df['high'] - df['low']
    df['Close-Open_Spread'] = df['close'] - df['open']
    df['Intraday_Momentum'] = df['High-Low_Spread'] - df['Close-Open_Spread']

    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = df['amount'] / df['volume']

    # Determine Volume Synchronization
    df['Log_Volume_Change'] = np.log(df['volume'] / df['volume'].shift(1))
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))

    # Integrate Price and Volume Dynamics
    df['Integrated_Indicator'] = df['Intraday_Momentum'] * df['Log_Return']
    df['Combined_Indicator'] = df['Integrated_Indicator'] + df['VWAP']

    # Enhance Factor with Short-Term Intraday and Relative Strength
    df['Short_Term_Intraday_Trend'] = (df['high'] - df['low']) / df['low']
    df['Rolling_Avg_Close'] = df['close'].rolling(window=10).mean()
    df['Short_Term_Relative_Strength'] = df['close'] / df['Rolling_Avg_Close']

    # Incorporate Advanced Volatility
    df['EMA_Realized_Volatility'] = (np.log(df['close'] / df['close'].shift(1))**2).ewm(span=20).mean().sqrt()

    # Incorporate Refined Liquidity
    volume_changes = (df['volume'] - df['volume'].shift(1)).fillna(0)  # Fill NaN with 0 for first row
    df['Volume_Volatility'] = (volume_changes**2).rolling(window=20).mean().sqrt()

    # Incorporate Market Sentiment
    df['Close_to_Open_Ratio'] = df['close'] / df['open'].shift(-1)

    # Final Alpha Factor
    df['Final_Alpha_Factor'] = (
        df['Short_Term_Intraday_Trend'] * 
        df['Combined_Indicator'] * 
        df['Short_Term_Relative_Strength'] * 
        df['EMA_Realized_Volatility'] * 
        df['Volume_Volatility'] * 
        df['Close_to_Open_Ratio']
    )

    return df['Final_Alpha_Factor']
