import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Total Volume and Total Dollar Value
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['close'] * df['volume']
    
    # Calculate Daily VWAP
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    df['short_term_momentum'] = df['vwap_deviation'].rolling(window=5).sum()
    df['cumulative_vwap_deviation'] += df['short_term_momentum']
    
    # Integrate Medium-Term Momentum (10 days)
    df['medium_term_momentum'] = df['vwap_deviation'].rolling(window=10).sum()
    df['cumulative_vwap_deviation'] += df['medium_term_momentum']
    
    # Integrate Long-Term Momentum (20 days)
    df['long_term_momentum'] = df['vwap_deviation'].rolling(window=20).sum()
    df['cumulative_vwap_deviation'] += df['long_term_momentum']
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['abs_vwap_deviation'] = abs(df['close'] - df['vwap'])
    df['intraday_volatility'] = df['high_low_range'] + df['abs_vwap_deviation']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['cumulative_vwap_deviation'] + df['intraday_volatility']
    
    return df['alpha_factor']
