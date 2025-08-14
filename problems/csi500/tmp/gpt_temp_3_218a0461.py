import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'].shift(-1) - df['open']) / df['open']
    
    # Weight by Volume
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Adjust by Sentiment
    df['sentiment_adjusted_return'] = df['volume_weighted_return'] * (1 + df['sentiment_score'])
    
    # Short-Term Lookback (5 days)
    short_term_ema = df['sentiment_adjusted_return'].ewm(span=5, min_periods=5).mean()
    short_term_std = df['sentiment_adjusted_return'].rolling(window=5, min_periods=5).std()
    
    # Mid-Term Lookback (10 days)
    mid_term_ema = df['sentiment_adjusted_return'].ewm(span=10, min_periods=10).mean()
    mid_term_std = df['sentiment_adjusted_return'].rolling(window=10, min_periods=10).std()
    
    # Long-Term Lookback (20 days)
    long_term_ema = df['sentiment_adjusted_return'].ewm(span=20, min_periods=20).mean()
    long_term_std = df['sentiment_adjusted_return'].rolling(window=20, min_periods=20).std()
    
    # Extra-Long Term Lookback (40 days)
    extra_long_term_ema = df['sentiment_adjusted_return'].ewm(span=40, min_periods=40).mean()
    extra_long_term_std = df['sentiment_adjusted_return'].rolling(window=40, min_periods=40).std()
    
    # Combine Multi-Period Volatilities
    combined_volatility = (
        0.4 * short_term_std + 
        0.3 * mid_term_std + 
        0.2 * long_term_std + 
        0.1 * extra_long_term_std
    )
    
    # Final Factor
    final_factor = df['sentiment_adjusted_return'] / combined_volatility
    
    return final_factor
