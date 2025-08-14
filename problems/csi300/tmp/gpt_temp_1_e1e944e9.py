import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Adjusted Close Price
    adj_close = df['close'] - df['open']
    
    # Intraday Momentum
    high_price_ind = df['high'] - df['open']
    low_price_ind = df['open'] - df['low']
    avg_intraday_mom = (high_price_ind + low_price_ind) / 2
    
    # Adjust for Dynamic Volume Impact on Intraday Momentum
    lookback_period = 10
    moving_avg_volume = df['volume'].rolling(window=lookback_period).mean()
    normalized_volume = df['volume'] / moving_avg_volume
    volume_impact_intraday_mom = avg_intraday_mom * normalized_volume
    
    # Incorporate Longer-Term Momentum Adjusted for Volume
    long_lookback_period = 20
    long_term_mom = (df['close'] - df['close'].shift(long_lookback_period)) / df['close'].shift(long_lookback_period)
    volume_ratio = df['volume'] / df['volume'].shift(long_lookback_period)
    vol_adj_long_term_mom = long_term_mom * (volume_ratio ** (1/3))
    
    # Consolidate All Factors
    total_factors = adj_close + volume_impact_intraday_mom + vol_adj_long_term_mom
    consolidated_factors = total_factors / df['open']
    
    # Daily Price Return
    daily_return = df['close'] - df['close'].shift(1)
    
    # Weighted Price Momentum
    positive_threshold = 0.01
    weighted_mom = daily_return.apply(lambda x: 1 if x > positive_threshold else 0)
    
    # Volume Spike Filter
    volume_spike_lookback = 7
    median_volume = df['volume'].rolling(window=volume_spike_lookback).median()
    volume_spike_indicator = df['volume'].apply(lambda x, y: 1 if x > y else 0, args=(median_volume,))
    
    # Apply Volume Spike Filter
    filtered_weighted_mom = weighted_mom * volume_spike_indicator
    
    # Final Alpha Factor Synthesis
    final_alpha_factor = (consolidated_factors + filtered_weighted_mom) * volume_spike_indicator
    
    return final_alpha_factor
