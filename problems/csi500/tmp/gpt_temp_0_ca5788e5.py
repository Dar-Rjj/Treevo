import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['dollar_value'] = df['close'] * df['volume']
    df['vwap'] = df.groupby(df.index.date)['dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum = df['close'].pct_change(5)
    df['short_term_factor'] = df['cumulative_vwap_deviation'] * short_term_momentum
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum = df['close'].pct_change(10)
    df['medium_term_factor'] = df['cumulative_vwap_deviation'] * medium_term_momentum
    
    # Final Alpha Factor
    df['alpha_factor'] = df['cumulative_vwap_deviation'] + df['short_term_factor'] + df['medium_term_factor']
    
    return df['alpha_factor']
