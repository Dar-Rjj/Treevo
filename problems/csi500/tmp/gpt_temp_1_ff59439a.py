import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['total_volume'] = df['volume']
    df['total_dollar_value'] = df['volume'] * df['close']
    df['vwap'] = df.groupby(df.index.date)['total_dollar_value'].cumsum() / df.groupby(df.index.date)['total_volume'].cumsum()
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - df['vwap']
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum_period = 5
    df['short_term_momentum'] = df['vwap_deviation'].rolling(window=short_term_momentum_period).sum()
    df['cumulative_vwap_deviation'] += df['short_term_momentum']
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum_period = 10
    df['medium_term_momentum'] = df['vwap_deviation'].rolling(window=medium_term_momentum_period).sum()
    df['cumulative_vwap_deviation'] += df['medium_term_momentum']
    
    # Integrate Long-Term Momentum (20 days)
    long_term_momentum_period = 20
    df['long_term_momentum'] = df['vwap_deviation'].rolling(window=long_term_momentum_period).sum()
    df['cumulative_vwap_deviation'] += df['long_term_momentum']
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = np.abs(df['close'] - df['vwap'])
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']
    
    # Adaptive Weighting
    df['historical_intraday_volatility'] = df['intraday_volatility'].rolling(window=long_term_momentum_period).mean()
    df['volatility_ratio'] = df['intraday_volatility'] / df['historical_intraday_volatility']
    
    # Define weights
    df['short_term_weight'] = np.where(df['volatility_ratio'] < 1, 0.4, 0.2)
    df['medium_term_weight'] = np.where(df['volatility_ratio'] < 1, 0.3, 0.4)
    df['long_term_weight'] = np.where(df['volatility_ratio'] < 1, 0.2, 0.3)
    df['intraday_volatility_weight'] = np.where(df['volatility_ratio'] < 1, 0.1, 0.1)
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        df['cumulative_vwap_deviation'] +
        df['short_term_momentum'] * df['short_term_weight'] +
        df['medium_term_momentum'] * df['medium_term_weight'] +
        df['long_term_momentum'] * df['long_term_weight'] +
        df['intraday_volatility'] * df['intraday_volatility_weight']
    )
    
    return df['alpha_factor']

# Example usage:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
