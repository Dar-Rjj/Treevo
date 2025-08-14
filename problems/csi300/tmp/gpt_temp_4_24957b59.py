import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['High_Low_Spread'] = df['high'] - df['low']
    df['Close_Open_Spread'] = df['close'] - df['open']
    df['Intraday_Momentum'] = (df['high'] - df['low']) - (df['close'] - df['open'])
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['VWAP'] = df['amount'] / df['volume']

    # Determine Volume Synchronization
    df['Log_Volume_Change'] = np.log(df['volume'] / df['volume'].shift(1))
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))

    # Integrate Price and Volume Dynamics
    df['Integrated_Indicator'] = df['Intraday_Momentum'] * df['Log_Return']
    df['Combined_Indicator'] = df['Integrated_Indicator'] + df['VWAP']

    # Enhance Factor with Intraday and Relative Strength
    df['Intraday_Trend'] = (df['high'] - df['low']) / df['low']
    df['Rolling_Avg_Close'] = df['close'].rolling(window=50).mean()
    df['Relative_Strength'] = df['close'] / df['Rolling_Avg_Close']

    # Incorporate Volatility
    log_returns = np.log(df['close'] / df['close'].shift(1))
    df['Advanced_Realized_Volatility'] = np.sqrt(log_returns.rolling(window=50).apply(lambda x: (x**2).mean()))

    # Incorporate Liquidity
    df['Average_Daily_Volume'] = df['volume'].rolling(window=50).mean()
    df['Volume_Volatility'] = np.sqrt(((df['volume'] - df['volume'].shift(1))**2).rolling(window=50).mean())

    # Incorporate Market Sentiment
    df['Close_to_Open_Ratio'] = df['close'] / df['open'].shift(-1)
    df['Open_to_Close_Ratio'] = df['open'] / df['close'].shift(1)

    # Final Alpha Factor
    df['Final_Alpha_Factor'] = (df['Intraday_Trend'] * 
                                df['Combined_Indicator'] * 
                                df['Relative_Strength'] * 
                                df['Advanced_Realized_Volatility'] * 
                                df['Volume_Volatility'] * 
                                df['Close_to_Open_Ratio'] * 
                                df['Open_to_Close_Ratio'])

    return df['Final_Alpha_Factor'].dropna()
