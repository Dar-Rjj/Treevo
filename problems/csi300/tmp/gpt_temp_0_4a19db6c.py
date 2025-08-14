import pandas as pd
import pandas as pd

def heuristics_v2(df):
    """
    Calculate the Price Momentum Oscillator.
    
    Parameters:
    - df: pandas DataFrame with columns (open, high, low, close, amount, volume) and index (date).
    
    Returns:
    - factor: pandas Series indexed by (date) representing the factor values.
    """
    # Calculate Long-Term Price Momentum
    long_term_high = df['high'].rolling(window=20).max()
    long_term_low = df['low'].rolling(window=20).min()
    long_term_momentum = long_term_high - long_term_low
    
    # Calculate Short-Term Price Momentum
    short_term_high = df['high'].rolling(window=5).max()
    short_term_low = df['low'].rolling(window=5).min()
    short_term_momentum = short_term_high - short_term_low
    
    # Derive the Oscillator
    oscillator = long_term_momentum - short_term_momentum
    
    # Interpret the absolute value of the oscillator
    factor = oscillator.abs().apply(lambda x: 'strong' if x > 1 else 'weak') + '_' + \
             oscillator.apply(lambda x: 'down' if x > 0 else 'up')
    
    return factor
