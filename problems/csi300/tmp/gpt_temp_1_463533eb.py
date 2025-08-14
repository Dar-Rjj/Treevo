import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Momentum metrics
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    df['Momentum_SMA_Diff'] = df['SMA_10'] - df['SMA_30']
    
    df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['Momentum_EMA_Diff'] = df['EMA_10'] - df['EMA_30']
    
    df['ROC_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Volatility metrics
    df['Volatility_Std_10'] = df['close'].pct_change().rolling(window=10).std()
    df['Volatility_Std_30'] = df['close'].pct_change().rolling(window=30).std()
    
    df['ATR_14'] = (df['high'] - df['low']).rolling(window=14).mean()
    
    df['Parkinson_Vol_30'] = np.sqrt((1 / (4 * np.log(2) * 30)) * ((np.log(df['high'] / df['low'])) ** 2).rolling(window=30).sum())
    
    # Volume-weighted metrics
    df['VWAP'] = (df['close'] * df['volume']) / df['volume']
    
    df['VWSMA_10'] = (df['close'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    
    df['VWEMA_10'] = (df['close'] * (2 / (10 + 1)) + df['VWEMA_10'].shift(1) * (1 - (2 / (10 + 1))) * (df['volume'] / df['volume'].shift(1))).fillna(method='bfill')
    
    # Liquidity metrics
    df['Turnover_Ratio_30'] = df['volume'].rolling(window=30).sum() / df['amount'].rolling(window=30).sum()
    
    df['Liquidity_Index_30'] = (df['volume'].rolling(window=30).sum() / df['amount'].rolling(window=30).sum()) * (df['high'] - df['low'])
    
    # Combine into final alpha factor
    df['Alpha_Factor'] = (df['Momentum_SMA_Diff'] + df['Momentum_EMA_Diff'] + df['ROC_10']) / (df['Volatility_Std_10'] + df['Volatility_Std_30'] + df['ATR_14'] + df['Parkinson_Vol_30']) * (df['VWAP'] + df['VWSMA_10'] + df['VWEMA_10']) / (df['Turnover_Ratio_30'] + df['Liquidity_Index_30'])
    
    return df['Alpha_Factor']
