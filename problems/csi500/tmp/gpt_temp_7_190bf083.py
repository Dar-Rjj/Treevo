import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Metrics
    df['intraday_high_low_spread'] = df['high'] - df['low']
    df['intraday_close_open_change'] = df['close'] - df['open']
    df['combined_intraday_movement'] = df['intraday_high_low_spread'] + df['intraday_close_open_change']
    df['volume_amount_weight'] = (df['volume'] + df['amount'])
    df['weighted_intraday_movement'] = df['combined_intraday_movement'] * df['volume_amount_weight']
    
    df['high_low_range'] = df['high'] - df['low']

    # Compute Momentum Metrics
    df['prev_high_low_range'] = df['high_low_range'].shift(1)
    df['close_to_close_return'] = (df['close'] / df['close'].shift(1)) - 1
    df['high_low_range_momentum'] = (df['high_low_range'] - df['prev_high_low_range']) * df['close_to_close_return']
    
    df['vol_adj_high_low_spread'] = df['high_low_range'] * df['volume_amount_weight']
    df['prev_vol_adj_high_low_spread'] = (df['prev_high_low_range'] * 
                                          (df['volume'].shift(1) + df['amount'].shift(1)))
    df['vol_adj_high_low_spread_momentum'] = (df['vol_adj_high_low_spread'] - 
                                              df['prev_vol_adj_high_low_spread'])

    # Aggregate and Adjust Factors
    df['combined_intraday_and_high_low'] = df['weighted_intraday_movement'] + df['high_low_range_momentum']
    
    df['volume_amount_change'] = df['volume_amount_weight'] - df['volume_amount_weight'].shift(1)
    df['adjusted_factor'] = df['combined_intraday_and_high_low'] + df['vol_adj_high_low_spread_momentum']
    
    df['final_factor'] = df.apply(lambda row: row['adjusted_factor'] * row['volume_amount_change'] 
                                  if row['volume_amount_change'] > 0 
                                  else row['adjusted_factor'] / abs(row['volume_amount_change']), axis=1)
    
    # Aggregate Spread Momentum over Rolling Window
    df['rolling_sum_final_factor'] = df['final_factor'].rolling(window=5).sum()
    
    return df['rolling_sum_final_factor']
