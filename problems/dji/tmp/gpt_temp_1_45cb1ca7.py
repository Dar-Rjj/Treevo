import pandas as pd

def heuristics_v2(df):
    # Compute the 5-day rolling standard deviation (volatility) of the close price as a weight adjuster
    df['std_5'] = df['close'].rolling(window=5).std()
    
    # Normalize the volatility to create a dynamic weight for each feature
    max_std = df['std_5'].max()
    df['weight_high'] = 1 / (1 + df['std_5'])
    df['weight_low'] = 1 - df['weight_high']
    df['volume_weight'] = (df['std_5'] / max_std) if max_std != 0 else 1
    
    # Compute the weighted sum of high, low, and volume to get the heuristic score
    df['heuristics_score'] = (df['high'] * df['weight_high']) + (df['low'] * df['weight_low']) + (df['volume'] * df['volume_weight'])
    
    # Drop the intermediate columns created for calculation
    heuristics_matrix = df.drop(columns=['std_5', 'weight_high', 'weight_low', 'volume_weight'])
    
    return heuristics_matrix
