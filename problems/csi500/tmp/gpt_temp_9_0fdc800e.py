import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df['total_dollar_value'].cumsum() / df['total_volume'].cumsum()
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_period = 5
    df['log_return_short'] = np.log(df['close'] / df['vwap'])
    df['weighted_momentum_short'] = (df['log_return_short'] * df['volume']).rolling(window=short_term_period).sum()
    df['cumulative_vwap_deviation_with_short'] = df['cumulative_vwap_deviation'] + df['weighted_momentum_short']
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_period = 10
    df['log_return_medium'] = np.log(df['close'] / df['vwap'])
    df['weighted_momentum_medium'] = (df['log_return_medium'] * df['volume']).rolling(window=medium_term_period).sum()
    df['cumulative_vwap_deviation_with_medium'] = df['cumulative_vwap_deviation_with_short'] + df['weighted_momentum_medium']
    
    # Integrate Long-Term Momentum (20 days)
    long_term_period = 20
    df['log_return_long'] = np.log(df['close'] / df['vwap'])
    df['weighted_momentum_long'] = (df['log_return_long'] * df['volume']).rolling(window=long_term_period).sum()
    df['cumulative_vwap_deviation_with_long'] = df['cumulative_vwap_deviation_with_medium'] + df['weighted_momentum_long']
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = np.abs(df['close'] - df['vwap'])
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']
    
    # Integrate Market Cap
    df['final_factor'] = df['cumulative_vwap_deviation_with_long'] + df['intraday_volatility'] * df['market_cap']
    
    return df['final_factor']
