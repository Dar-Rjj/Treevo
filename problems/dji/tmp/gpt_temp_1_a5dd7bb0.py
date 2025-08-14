import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Weighted Close Price
    df['weighted_close'] = df['close'] * df['volume']
    
    # Calculate Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Volatility
    df['volatility'] = (df['high'] - df['low']) / df['close']
    
    # Calculate Momentum
    momentum_window = 10
    df['momentum'] = df['weighted_close'].pct_change(momentum_window)
    
    # Adjust Momentum by Price Range and Volatility
    df['adjusted_momentum'] = df['momentum'] * df['price_range'] * df['volatility']
    
    # Integrate Volume Trend and Volume-Price Correlation
    # Calculate Volume Change
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate Volume-Price Correlation
    correlation_window = 30
    df['volume_price_corr'] = df[['close', 'volume']].rolling(window=correlation_window).corr().iloc[::2, -1].reset_index(drop=True)
    
    # Adjust Final Momentum
    df['final_momentum'] = df['adjusted_momentum'] * df['volume_change'] * df['volume_price_corr']
    
    # Incorporate Price Trend Analysis
    # Calculate Short-Term Moving Average
    short_term_window = 5
    df['short_term_ma'] = df['close'].rolling(window=short_term_window).mean()
    
    # Calculate Long-Term Moving Average
    long_term_window = 20
    df['long_term_ma'] = df['close'].rolling(window=long_term_window).mean()
    
    # Determine Trend Direction
    df['trend_direction'] = np.where(df['short_term_ma'] > df['long_term_ma'], 1, -1)
    
    # Adjust Final Alpha Factor
    df['alpha_factor'] = df['final_momentum'] * df['trend_direction']
    
    return df['alpha_factor']
