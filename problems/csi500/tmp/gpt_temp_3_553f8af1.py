import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['dollar_value'] = df['volume'] * df['close']
    daily_vwap = (df.groupby(df.index.date)['dollar_value'].sum() / 
                  df.groupby(df.index.date)['volume'].sum()).reindex(df.index, method='ffill')
    
    # Calculate VWAP Deviation
    vwap_deviation = df['close'] - daily_vwap
    
    # Calculate Cumulative VWAP Deviation
    cumulative_vwap_deviation = vwap_deviation.cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum = vwap_deviation.rolling(window=5).sum()
    cumulative_vwap_deviation += short_term_momentum
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum = vwap_deviation.rolling(window=10).sum()
    cumulative_vwap_deviation += medium_term_momentum
    
    # Integrate Long-Term Momentum (20 days)
    long_term_momentum = vwap_deviation.rolling(window=20).sum()
    cumulative_vwap_deviation += long_term_momentum
    
    # Calculate Intraday Volatility
    high_low_range = df['high'] - df['low']
    absolute_vwap_deviation = abs(vwap_deviation)
    intraday_volatility = high_low_range + absolute_vwap_deviation
    
    # Final Alpha Factor
    alpha_factor = (cumulative_vwap_deviation + 
                    short_term_momentum + 
                    medium_term_momentum + 
                    long_term_momentum + 
                    intraday_volatility)
    
    return alpha_factor
