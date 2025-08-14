import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Adaptive Short-Term Momentum
    short_term_period = 5
    df['short_term_momentum'] = df['vwap_deviation'].rolling(window=short_term_period).sum()
    df['cumulative_vwap_deviation'] += df['short_term_momentu']

    # Integrate Adaptive Medium-Term Momentum
    medium_term_period = 10
    df['medium_term_momentum'] = df['vwap_deviation'].rolling(window=medium_term_period).sum()
    df['cumulative_vwap_deviation'] += df['medium_term_momentum']
    
    # Integrate Adaptive Long-Term Momentum
    long_term_period = 20
    df['long_term_momentum'] = df['vwap_deviation'].rolling(window=long_term_period).sum()
    df['cumulative_vwap_deviation'] += df['long_term_momentum']
    
    # Calculate VWAP Deviation Volatility
    df['vwap_deviation_volatility'] = df['vwap_deviation'].rolling(window=20).std()
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = df['vwap_deviation'].abs()
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['cumulative_vwap_deviation'] 
                                + df['short_term_momentum'] 
                                + df['medium_term_momentum'] 
                                + df['long_term_momentum'] 
                                + df['vwap_deviation_volatility'] 
                                + df['intraday_volatility'])
    
    return df['final_alpha_factor']
