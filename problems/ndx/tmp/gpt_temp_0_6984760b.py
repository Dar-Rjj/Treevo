import pandas as pd
import pandas as pd

def heuristics(df):
    # Calculate Short-Term Return
    short_term_return = df['close'].pct_change(periods=5)
    
    # Calculate Long-Term Return
    long_term_return = df['close'].pct_change(periods=20)
    
    # Calculate Volume-Weighted Short-Term Return
    volume_weighted_short_return = df['volume'] * short_term_return
    
    # Calculate Volume-Weighted Long-Term Return
    volume_weighted_long_return = df['volume'] * long_term_return
    
    # Calculate Short-Term Volatility (Average True Range over 5 days)
    true_range = df[['high', 'low']].rolling(window=5).apply(lambda x: max(x['high']) - min(x['low']), raw=True)
    short_term_volatility = true_range.mean(axis=1)
    
    # Adjust for Volatility
    adjusted_volatility = (volume_weighted_short_return / short_term_volatility) - volume_weighted_long_return
    
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Close-to-Close Return
    close_to_close_return = df['close'].pct_change()
    
    # Combine Intraday Volatility and Close-to-Close Return
    combined_intraday_close = intraday_volatility * close_to_close_return
    
    # Calculate Open-to-Close Return
    open_to_close_return = (df['close'] / df['open']) - 1
    
    # Calculate Open-to-High Ratio
    open_to_high_ratio = df['high'] / df['open']
    
    # Combine Open-to-Close Return and Open-to-High Ratio
    combined_open_close_high = open_to_close_return * open_to_high_ratio
    
    # Final Adjustment
    final_factor = adjusted_volatility + combined_intraday_close + combined_open_close_high
    
    return final_factor.dropna()
