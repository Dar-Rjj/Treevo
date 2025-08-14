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
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].rolling(window=len(df), min_periods=1).sum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum = df['vwap_deviation'].rolling(window=5, min_periods=1).sum()
    df['short_term_combined'] = 0.4 * df['cumulative_vwap_deviation'] + 0.6 * short_term_momentum
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum = df['vwap_deviation'].rolling(window=10, min_periods=1).sum()
    df['medium_term_combined'] = 0.3 * df['cumulative_vwap_deviation'] + 0.7 * medium_term_momentum
    
    # Integrate Long-Term Momentum (20 days)
    long_term_momentum = df['vwap_deviation'].rolling(window=20, min_periods=1).sum()
    df['long_term_combined'] = 0.2 * df['cumulative_vwap_deviation'] + 0.8 * long_term_momentum
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = df['vwap_deviation'].abs()
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        0.3 * df['cumulative_vwap_deviation'] +
        0.2 * df['short_term_combined'] +
        0.2 * df['medium_term_combined'] +
        0.2 * df['long_term_combined'] +
        0.1 * df['intraday_volatility']
    )
    
    return df['alpha_factor']
