import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df.groupby(level=0)['total_dollar_value'].sum() / df.groupby(level=0)['total_volume'].sum()
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df.groupby(level=0)['vwap_deviation'].transform('cumsum')
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum_period = 5
    df['short_term_momentum'] = df['close'].pct_change(periods=short_term_momentum_period)
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum_period = 10
    df['medium_term_momentum'] = df['close'].pct_change(periods=medium_term_momentum_period)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['cumulative_vwap_deviation'] + df['short_term_momentum'] + df['medium_term_momentum']
    
    return df['alpha_factor'].dropna()
