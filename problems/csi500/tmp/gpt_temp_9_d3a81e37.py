import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].transform('sum') / df.groupby(df.index.date)['total_volume'].transform('sum')
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Dynamic Short-Term Momentum
    short_term_period = 5  # Adjustable based on market conditions
    df['short_term_momentum'] = df['vwap_deviation'].rolling(window=short_term_period).sum()
    df['cumulative_vwap_deviation_with_short'] = df['cumulative_vwap_deviation'] + df['short_term_momentum']
    
    # Integrate Dynamic Medium-Term Momentum
    medium_term_period = 10  # Adjustable based on market conditions
    df['medium_term_momentum'] = df['vwap_deviation'].rolling(window=medium_term_period).sum()
    df['cumulative_vwap_deviation_with_medium'] = df['cumulative_vwap_deviation_with_short'] + df['medium_term_momentum']
    
    # Integrate Dynamic Long-Term Momentum
    long_term_period = 20  # Adjustable based on market conditions
    df['long_term_momentum'] = df['vwap_deviation'].rolling(window=long_term_period).sum()
    df['cumulative_vwap_deviation_with_long'] = df['cumulative_vwap_deviation_with_medium'] + df['long_term_momentum']
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = (df['close'] - df['vwap']).abs()
    df['intraday_volatility'] = 0.5 * df['high_low_range'] + 0.5 * df['absolute_vwap_deviation']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['cumulative_vwap_deviation_with_long'] + df['intraday_volatility']
    
    return df['alpha_factor']
