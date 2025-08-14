import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()
    
    # Calculate Weighted Moving Averages
    short_term_weights = [2, 3, 4, 5, 6]
    long_term_weights = list(range(2, 21))
    
    def wma(series, weights):
        return (series * weights).sum() / sum(weights)
    
    df['short_term_wma'] = df['price_change'].rolling(window=5).apply(lambda x: wma(x, short_term_weights), raw=False)
    df['long_term_wma'] = df['price_change'].rolling(window=20).apply(lambda x: wma(x, long_term_weights), raw=False)
    
    # Compute Momentum Oscillator
    df['momentum_oscillator'] = df['short_term_wma'] - df['long_term_wma']
    
    # Integrate Trading Volume
    df['volume_change'] = df['volume'].diff()
    volume_short_term_weights = [1, 2, 3, 4, 5]
    volume_long_term_weights = list(range(1, 21))
    
    df['short_term_volume_wma'] = df['volume_change'].rolling(window=5).apply(lambda x: wma(x, volume_short_term_weights), raw=False)
    df['long_term_volume_wma'] = df['volume_change'].rolling(window=20).apply(lambda x: wma(x, volume_long_term_weights), raw=False)
    
    df['adjusted_momentum_oscillator'] = df['momentum_oscillator'] + (df['short_term_volume_wma'] - df['long_term_volume_wma'])
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Adjust Intraday Return by Intraday Range
    df['adjusted_intraday_return'] = df['intraday_return'] / df['intraday_range']
    
    # Incorporate Volume Trend
    df['volume_trend_adjusted_intraday_return'] = df['adjusted_intraday_return'] * df['volume_change']
    
    # Calculate Volume-Weighted Average Price (VWAP)
    df['vwap'] = ((df['high'] + df['low']) / 2 * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate Daily Return Using VWAP
    df['daily_return_vwap'] = (df['close'] - df['vwap']) / df['vwap']
    
    # Smooth the Daily Return
    df['smoothed_return'] = df['daily_return_vwap'].ewm(span=10, adjust=False).mean() * df['volume']
    
    # Factor Calculation
    df['factor_calculation'] = df['volume_trend_adjusted_intraday_return'] + df['intraday_range']
    
    # Calculate Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['open']
    
    # Incorporate Intraday Volatility
    df['factor_calculation'] = df['factor_calculation'] / df['intraday_volatility']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['adjusted_momentum_oscillator'] + df['factor_calculation']
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['price_change']
    
    # Combine Alpha Factors
    df['final_alpha_factor'] = df['final_alpha_factor'] * df['smoothed_return']
    
    return df['final_alpha_factor'].dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# alpha_factor = heuristics_v2(df)
