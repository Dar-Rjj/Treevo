import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['dollar_value'] = df['close'] * df['volume']
    df['daily_vwap'] = df.groupby(df.index.date)['dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['daily_vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df.groupby(df.index.date)['vwap_deviation'].cumsum()
    
    # Integrate Dynamic Multi-Period Momentum
    short_term_momentum = df['vwap_deviation'].rolling(window=5).sum()
    medium_term_momentum = df['vwap_deviation'].rolling(window=10).sum()
    long_term_momentum = df['vwap_deviation'].rolling(window=20).sum()
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['abs_vwap_deviation'] = (df['close'] - df['daily_vwap']).abs()
    df['intraday_volatility'] = df['high_low_range'] + df['abs_vwap_deviation']
    
    # Adjust Momentums for Volatility
    short_term_momentum_adjusted = short_term_momentum / df['intraday_volatility']
    medium_term_momentum_adjusted = medium_term_momentum / df['intraday_volatility']
    long_term_momentum_adjusted = long_term_momentum / df['intraday_volatility']
    
    # Combine Adjusted Momentums with Cumulative VWAP Deviation
    df['alpha_factor'] = (df['cumulative_vwap_deviation'] 
                          + short_term_momentum_adjusted 
                          + medium_term_momentum_adjusted 
                          + long_term_momentum_adjusted 
                          + df['intraday_volatility'])
    
    return df['alpha_factor']
