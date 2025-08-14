import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Overnight Return
    df['overnight_return'] = (df['open'].shift(-1) - df['close']) / df['close']
    
    # Compute Rolling Average of Intraday Returns (5 days)
    df['rolling_intraday_return'] = df['intraday_return'].rolling(window=5).mean()
    
    # Compute Rolling Average of Overnight Returns (5 days)
    df['rolling_overnight_return'] = df['overnight_return'].rolling(window=5).mean()
    
    # Calculate Intraday Momentum
    momentum_factor = 1.2
    df['intraday_momentum'] = df['rolling_intraday_return'] * momentum_factor
    
    # Calculate Overnight Reversal
    reversal_factor = -0.8
    df['overnight_reversal'] = df['rolling_overnight_return'] * reversal_factor
    
    # Integrate Market Volatility
    df['true_range'] = df[['high' - 'low', abs('high' - 'close').shift(1), abs('low' - 'close').shift(1)]].max(axis=1)
    df['rolling_volatility'] = df['true_range'].rolling(window=20).mean()
    df['volatility_adjustment'] = np.where(df['rolling_volatility'] > df['rolling_volatility'].mean(), 0.9, 1.1)
    
    # Integrate Market Liquidity
    lookback_period = 10
    df['average_volume'] = df['volume'].rolling(window=lookback_period).mean()
    df['liquidity_adjustment'] = np.where(df['average_volume'] < df['volume'].mean(), 0.8, 1.2)
    
    # Combine Intraday Momentum, Overnight Reversal, and Market Dynamics
    df['composite_factor'] = (df['intraday_momentum'] + 
                              df['overnight_reversal']) * 
                             df['volatility_adjustment'] * 
                             df['liquidity_adjustment']
    
    return df['composite_factor']
