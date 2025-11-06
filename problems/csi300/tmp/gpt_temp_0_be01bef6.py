import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Multi-Period Trend Acceleration
    # Short-term: 3-day slope / 5-day slope
    df['short_slope_3'] = df['close'].rolling(window=3).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 2 if len(x) == 3 else np.nan, raw=False)
    df['short_slope_5'] = df['close'].rolling(window=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 4 if len(x) == 5 else np.nan, raw=False)
    df['short_accel'] = df['short_slope_3'] / df['short_slope_5']
    
    # Medium-term: 5-day slope / 10-day slope
    df['medium_slope_5'] = df['close'].rolling(window=5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 4 if len(x) == 5 else np.nan, raw=False)
    df['medium_slope_10'] = df['close'].rolling(window=10).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / 9 if len(x) == 10 else np.nan, raw=False)
    df['medium_accel'] = df['medium_slope_5'] / df['medium_slope_10']
    
    # Volume-Price Divergence
    # Short-term: sign(5-day price return - 5-day volume return)
    df['price_return_5'] = df['close'].pct_change(5)
    df['volume_return_5'] = df['volume'].pct_change(5)
    df['short_div'] = np.sign(df['price_return_5'] - df['volume_return_5'])
    
    # Medium-term: sign(10-day price return - 10-day volume return)
    df['price_return_10'] = df['close'].pct_change(10)
    df['volume_return_10'] = df['volume'].pct_change(10)
    df['medium_div'] = np.sign(df['price_return_10'] - df['volume_return_10'])
    
    # Generate Alignment Scores
    # Short-term: +2 if accel>1 & div=+1, -2 if accel<1 & div=-1, else 0
    df['short_alignment'] = 0
    df.loc[(df['short_accel'] > 1) & (df['short_div'] == 1), 'short_alignment'] = 2
    df.loc[(df['short_accel'] < 1) & (df['short_div'] == -1), 'short_alignment'] = -2
    
    # Medium-term: +2 if accel>1 & div=+1, -2 if accel<1 & div=-1, else 0
    df['medium_alignment'] = 0
    df.loc[(df['medium_accel'] > 1) & (df['medium_div'] == 1), 'medium_alignment'] = 2
    df.loc[(df['medium_accel'] < 1) & (df['medium_div'] == -1), 'medium_alignment'] = -2
    
    # Composite Factor
    df['total_alignment'] = df['short_alignment'] + df['medium_alignment']
    df['abs_return_5'] = df['price_return_5'].abs()
    df['abs_return_10'] = df['price_return_10'].abs()
    df['return_magnitude'] = (df['abs_return_5'] + df['abs_return_10']) / 2
    
    # Final factor
    factor = df['total_alignment'] * df['return_magnitude']
    
    return factor
