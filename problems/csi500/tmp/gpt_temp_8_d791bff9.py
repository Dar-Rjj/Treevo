import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Average Intraday Return
    df['avg_intraday_return'] = df['intraday_return'].rolling(window=20).mean()
    
    # Calculate Intraday Volatility
    df['squared_intraday_return'] = df['intraday_return'] ** 2
    df['intraday_volatility'] = df['squared_intraday_return'].rolling(window=20).sum() ** 0.5
    
    # Calculate Moving Average of Intraday Volatility
    df['moving_avg_intraday_volatility'] = df['intraday_volatility'].rolling(window=20).mean()
    
    # Weight by Volume
    df['volume_weighted_volatility'] = df['moving_avg_intraday_volatility'] * df['volume']
    
    # Incorporate Short-Term Price Momentum
    df['close_5d_sma'] = df['close'].rolling(window=5).mean()
    df['short_term_momentum'] = df['close'] - df['close_5d_sma']
    
    # Adjust for Intraday Trend
    df['10d_high'] = df['high'].rolling(window=10).max()
    df['10d_low'] = df['low'].rolling(window=10).min()
    df['intraday_trend'] = (df['10d_high'] - df['10d_low']) * df['avg_intraday_return']
    
    # Incorporate Long-Term Trend
    df['close_100d_sma'] = df['close'].rolling(window=100).mean()
    df['long_term_trend'] = df['close'] - df['close_100d_sma']
    
    # Adaptive Smoothing
    df['ema_intraday_volatility'] = df['intraday_volatility'].ewm(span=20, adjust=False).mean()
    df['volume_weighted_intraday_momentum'] = df['volume'] * (df['close'] - df['close_5d_sma'])
    df['ema_volume_weighted_momentum'] = df['volume_weighted_intraday_momentum'].ewm(span=20, adjust=False).mean()
    
    # Integrate Macroeconomic Indicators
    # Assuming these are available in the DataFrame
    df['macroeconomic_impact'] = -df['unemployment_rate'] + df['cpi'] - df['interest_rates']
    
    # Final Alpha Factor
    df['alpha_factor'] = (
        df['volume_weighted_volatility'] +
        df['short_term_momentum'] +
        df['intraday_trend'] +
        df['long_term_trend'] +
        df['macroeconomic_impact']
    )
    
    return df['alpha_factor'].dropna()

# Example usage:
# alpha_factor = heuristics_v2(your_dataframe)
