import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['High-Low_Spread'] = df['high'] - df['low']
    df['Close-Open_Spread'] = df['close'] - df['open']
    df['Intraday_Momentum'] = df['High-Low_Spread'] - df['Close-Open_Spread']
    
    # Calculate VWAP
    df['VWAP'] = df['amount'] / df['volume']
    
    # Determine Volume Synchronization
    df['Log_Volume_Change'] = np.log(df['volume'] / df['volume'].shift(1))
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Integrate Price and Volume Dynamics
    df['Integrated_Indicator'] = df['Intraday_Momentum'] * df['Log_Return']
    df['Combined_Indicator'] = df['Integrated_Indicator'] + df['VWAP']
    
    # Enhance Factor with Intraday and Relative Strength
    df['Intraday_Trend'] = (df['high'] - df['low']) / df['low']
    df['Rolling_Average_Close'] = df['close'].rolling(window=10).mean().shift(1)
    df['Relative_Strength'] = df['close'] / df['Rolling_Average_Close']
    
    # Incorporate Advanced Volatility
    log_returns = np.log(df['close'] / df['close'].shift(1))
    df['Advanced_Realized_Volatility'] = np.sqrt(log_returns.rolling(window=10).apply(lambda x: (x**2).mean()))
    
    # Incorporate Refined Liquidity
    volume_changes = df['volume'] - df['volume'].shift(1)
    df['Volume_Volatility'] = np.sqrt(volume_changes.rolling(window=10).apply(lambda x: (x**2).mean()))
    df['Volume_Imbalance'] = np.abs(df['volume'] - df['volume'].shift(1))
    
    # Incorporate Enhanced Market Sentiment
    df['Close-to-Open_Ratio'] = df['close'] / df['open'].shift(-1)
    df['Open-to-Close_Ratio'] = df['open'] / df['close']
    
    # Final Alpha Factor
    df['Final_Alpha_Factor'] = (
        df['Intraday_Trend'] * 
        df['Combined_Indicator'] * 
        df['Relative_Strength'] * 
        df['Advanced_Realized_Volatility'] * 
        (df['Volume_Volatility'] + df['Volume_Imbalance']) * 
        (df['Close-to-Open_Ratio'] + df['Open-to-Close_Ratio'])
    )
    
    return df['Final_Alpha_Factor'].dropna()
