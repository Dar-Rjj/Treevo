import pandas as pd

def heuristics_v2(df):
    # Calculate short-term and long-term exponential moving averages
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate the smoothed price-volume ratio
    pv_ratio_smoothed = (df['close'] * df['volume']).ewm(span=10, adjust=False).mean() / df['volume'].ewm(span=10, adjust=False).mean()
    
    # Compute the difference between short and long exponential moving averages
    ema_diff = short_ema - long_ema
    
    # Apply a custom heuristic to combine the EMA difference and smoothed price-volume ratio
    combined_factor = ema_diff + pv_ratio_smoothed
    
    # Apply a logarithmic transformation to the combined factor
    log_transformed_factor = combined_factor.apply(lambda x: x if x <= 0 else math.log(1 + x))
    
    # Rank the stocks based on the transformed factor
    heuristics_matrix = log_transformed_factor.rank(pct=True)
    
    return heuristics_matrix
