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
    
    # Integrate Multi-Period Momentum
    df['short_term_momentum'] = df['close'].ewm(span=5, adjust=False).mean()
    df['long_term_momentum'] = df['close'].ewm(span=20, adjust=False).mean()
    df['momentum'] = df['short_term_momentum'] - df['long_term_momentum']
    
    # Analyze Volume Trends
    df['daily_volume_change'] = df['volume'] - df['volume'].shift(1)
    df['ema_daily_volume'] = df['volume'].ewm(span=10, adjust=False).mean()
    df['volume_trend'] = np.where(df['volume'] > df['ema_daily_volume'], 1, -1)
    
    # Combine Factors for Final Alpha Factor
    df['alpha_factor'] = (df['cumulative_vwap_deviation'] + 
                          df['momentum'] + 
                          df['volume_trend'])
    
    return df['alpha_factor']
