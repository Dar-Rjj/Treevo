import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.special import expit

def heuristics_v2(df):
    # Calculate Intraday Price Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Compute Intraday Return
    df['intraday_return'] = df['close'] - df['open']
    
    # Determine Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    
    # Adjust Volatility by Volume
    df['adjusted_volatility'] = df['intraday_volatility'] * df['volume']
    
    # Calculate Short-Term Momentum
    df['short_term_momentum'] = df['close'].diff(1)
    
    # Calculate Long-Term Momentum
    df['long_term_momentum'] = df['close'].diff(5)
    
    # Integrate Additional Market Data
    df['avg_volume_5d'] = df['volume'].rolling(window=5).mean()
    df['relative_volume'] = df['volume'] / df['avg_volume_5d']
    
    # Calculate Money Flow Index (MFI)
    def calculate_mfi(high, low, close, volume, window=14):
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_money_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_money_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        mfi = 100 - (100 / (1 + (positive_money_flow.rolling(window=window).sum() / negative_money_flow.rolling(window=window).sum())))
        return mfi
    
    df['mfi'] = calculate_mfi(df['high'], df['low'], df['close'], df['volume'])
    
    # Refine Momentum Calculations
    df['smoothed_short_term_momentum'] = df['short_term_momentum'].ewm(span=2).mean()
    df['smoothed_long_term_momentum'] = df['long_term_momentum'].ewm(span=6).mean()
    
    # Optimize Volatility Adjustments
    df['optimized_adjusted_volatility'] = expit(df['adjusted_volatility'])
    
    # Combine Intraday Return, Adjusted Volatility, Momentum, and Additional Indicators
    df['combined_value'] = (
        df['intraday_return'] 
        - df['optimized_adjusted_volatility'] 
        + df['smoothed_short_term_momentum'] 
        + df['smoothed_long_term_momentum'] 
        + df['relative_volume'] 
        + df['mfi']
    )
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = np.where(df['combined_value'] > 0, 1, 0)
    
    return df['alpha_factor']
