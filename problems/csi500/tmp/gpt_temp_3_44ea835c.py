import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].cumsum() / df.groupby(df.index.date)['total_volume'].cumsum()
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum = df['vwap_deviation'].rolling(window=5).sum()
    df['short_term_momentum'] = short_term_momentum
    df['cumulative_vwap_deviation'] += df['short_term_momentum']
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum = df['vwap_deviation'].rolling(window=10).sum()
    df['medium_term_momentum'] = medium_term_momentum
    df['cumulative_vwap_deviation'] += df['medium_term_momentum']
    
    # Integrate Long-Term Momentum (20 days)
    long_term_momentum = df['vwap_deviation'].rolling(window=20).sum()
    df['long_term_momentum'] = long_term_momentum
    df['cumulative_vwap_deviation'] += df['long_term_momentum']
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = np.abs(df['close'] - df['vwap'])
    df['intraday_volatility'] = np.sqrt((df['high_low_range'] + df['absolute_vwap_deviation']).rolling(window=20).sum())
    
    # Final Alpha Factor
    df['alpha_factor'] = df['cumulative_vwap_deviation'] + 0.5 * df['intraday_volatility']
    
    return df['alpha_factor']
