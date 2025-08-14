import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term and Long-Term Moving Averages
    short_term_ma = df['close'].rolling(window=5).mean()
    long_term_ma = df['close'].rolling(window=20).mean()
    
    # Compute Momentum Score
    momentum_score = short_term_ma - long_term_ma
    
    # Volume Confirmation
    volume_ratio = df['volume'].rolling(window=5).sum() / df['volume'].rolling(window=20).sum()
    volume_threshold = 1.5
    final_momentum_factor = momentum_score * (1.5 if volume_ratio > volume_threshold else 1)
    
    # Calculate Intraday Range
    high_to_low_diff = df['high'] - df['low']
    
    # Adjust Close-to-Open Return by Intraday Volatility
    close_to_open_return = (df['close'] - df['open']) / df['open']
    adjusted_return = close_to_open_return / high_to_low_diff
    
    # Combine Momentum and Volatility Factors
    combined_alpha_factor = final_momentum_factor * adjusted_return
    
    return combined_alpha_factor
