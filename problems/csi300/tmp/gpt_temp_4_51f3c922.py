import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Close-Weighted by Volume for short-term and long-term MAs
    df['Close_Weighted'] = df['close'] * df['volume']

    # Short-Term (e.g., 5 days) Volume Weighted Moving Average
    short_term_vol_weighted_ma = df['Close_Weighted'].rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # Long-Term (e.g., 20 days) Volume Weighted Moving Average
    long_term_vol_weighted_ma = df['Close_Weighted'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

    # Generate Crossover Signal
    crossover_signal = short_term_vol_weighted_ma - long_term_vol_weighted_ma

    # Calculate Close Price Moving Average for the last N days
    close_ma = df['close'].rolling(window=20).mean()

    # Initialize Cumulative Sum to 0
    cumulative_sum = 0
    cumulative_diffs = []

    for i, row in df.iterrows():
        # Subtract Moving Average from Close Price
        diff = row['close'] - close_ma.loc[i]
        
        # Multiply Difference by Daily Volume
        vol_weighted_diff = diff * row['volume']
        
        # Add to Cumulative Sum
        cumulative_sum += vol_weighted_diff
        cumulative_diffs.append(cumulative_sum)
    
    df['cumulative_vol_weighted_diff'] = cumulative_diffs
    
    # Combine both factors into a single alpha factor
    alpha_factor = crossover_signal + df['cumulative_vol_weighted_diff']
    
    return alpha_factor
