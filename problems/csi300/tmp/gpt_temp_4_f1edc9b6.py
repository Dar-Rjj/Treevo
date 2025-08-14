import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Forward Close-to-Open Return
    df['fwd_c_to_o'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['vol_weighted_return'] = df['volume'] * df['fwd_c_to_o']

    # Simplified Volatility Adjustment: Calculate True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], 
                      abs(x['high'] - x['close'].shift(1)), 
                      abs(x['low'] - x['close'].shift(1))), 
        axis=1
    )

    # Determine the volatility multiplier
    tr_mean = df['true_range'].mean()
    df['vol_multiplier'] = (tr_mean / df['true_range']).clip(lower=0.5, upper=2.0)

    # Adjust Volume-Weighted Return with Volatility
    df['adjusted_vol_weighted_return'] = df['vol_weighted_return'] * df['vol_multiplier']

    return df['adjusted_vol_weighted_return']
