import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volume-Adjusted Gap Momentum factor that combines gap returns with volume trends
    and volume persistence patterns to predict future stock returns.
    """
    df = data.copy()
    
    # Gap Return Component
    df['gap_return'] = df['open'] / df['close'].shift(1) - 1
    
    # Filter extreme gaps using rolling 60-day statistics
    gap_std = df['gap_return'].rolling(window=60, min_periods=30).std()
    gap_mean = df['gap_return'].rolling(window=60, min_periods=30).mean()
    gap_threshold = 3 * gap_std
    
    # Apply filtering - keep gaps within 3 standard deviations
    filtered_gap = df['gap_return'].copy()
    extreme_mask = (df['gap_return'] - gap_mean).abs() > gap_threshold
    filtered_gap[extreme_mask] = np.sign(df['gap_return'][extreme_mask]) * gap_threshold[extreme_mask]
    
    # Volume Trend Component
    df['volume_trend_1d'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_trend_5d'] = df['volume'] / df['volume'].shift(5) - 1
    
    # Volume Persistence
    df['volume_increase'] = df['volume'] > df['volume'].shift(1)
    df['volume_decrease'] = df['volume'] < df['volume'].shift(1)
    
    # Calculate consecutive increases
    df['consec_increase'] = 0
    inc_count = 0
    for i in range(len(df)):
        if df['volume_increase'].iloc[i]:
            inc_count += 1
        else:
            inc_count = 0
        df.iloc[i, df.columns.get_loc('consec_increase')] = inc_count
    
    # Calculate consecutive decreases
    df['consec_decrease'] = 0
    dec_count = 0
    for i in range(len(df)):
        if df['volume_decrease'].iloc[i]:
            dec_count += 1
        else:
            dec_count = 0
        df.iloc[i, df.columns.get_loc('consec_decrease')] = dec_count
    
    # Volume persistence score (positive for increases, negative for decreases)
    df['volume_persistence'] = np.where(
        df['consec_increase'] > df['consec_decrease'],
        df['consec_increase'],
        -df['consec_decrease']
    )
    
    # Combined Factor
    # Gap-Volume Interaction terms
    df['gap_volume_trend'] = filtered_gap * df['volume_trend_5d']
    df['gap_volume_persistence'] = filtered_gap * df['volume_persistence']
    
    # Main combined factor (weighted combination)
    df['combined_factor'] = (
        0.6 * df['gap_volume_trend'] + 
        0.4 * df['gap_volume_persistence']
    )
    
    # Forward returns for validation (not used in final signal, just for reference)
    df['forward_5d_return'] = df['close'].shift(-5) / df['close'] - 1
    df['forward_10d_return'] = df['close'].shift(-10) / df['close'] - 1
    
    # Final signal generation - rank by combined factor
    df['factor_rank'] = df['combined_factor'].rank(pct=True)
    
    return df['factor_rank']
