import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    df['daily_vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['daily_vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].groupby(df.index.date).cumsum()
    
    # Integrate Multi-Period Momentum
    short_term_momentum = df['vwap_deviation'].rolling(window=5).sum()
    medium_term_momentum = df['vwap_deviation'].rolling(window=10).sum()
    long_term_momentum = df['vwap_deviation'].rolling(window=20).sum()
    
    df['multi_period_momentum'] = short_term_momentum + medium_term_momentum + long_term_momentum
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['abs_vwap_deviation'] = (df['close'] - df['daily_vwap']).abs()
    df['intraday_volatility'] = df['high_low_range'] + df['abs_vwap_deviation']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['cumulative_vwap_deviation'] + df['multi_period_momentum'] + df['intraday_volatility']
    
    return df['alpha_factor']
