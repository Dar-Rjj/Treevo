import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'].shift(-1) / df['open'].shift(1) - 1
    
    # Weight by Volume
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Dynamic Lookback Periods
    short_term_lookback = 5
    mid_term_lookback = 10
    long_term_lookback = 20
    
    df['short_ma_vol_weighted_return'] = df['volume_weighted_return'].rolling(window=short_term_lookback).mean()
    df['mid_ma_vol_weighted_return'] = df['volume_weighted_return'].rolling(window=mid_term_lookback).mean()
    df['long_ma_vol_weighted_return'] = df['volume_weighted_return'].rolling(window=long_term_lookback).mean()
    
    df['short_std_vol_weighted_return'] = df['volume_weighted_return'].rolling(window=short_term_lookback).std()
    df['mid_std_vol_weighted_return'] = df['volume_weighted_return'].rolling(window=mid_term_lookback).std()
    df['long_std_vol_weighted_return'] = df['volume_weighted_return'].rolling(window=long_term_lookback).std()
    
    # Integrate Market Trend
    df['market_momentum'] = df['close_market_index'] - df['close_market_index'].shift(1)
    
    # Adjust Factor Based on Market Momentum
    short_weight = (df['market_momentum'] > 0) * 0.6 + (df['market_momentum'] <= 0) * 0.3
    long_weight = (df['market_momentum'] > 0) * 0.3 + (df['market_momentum'] <= 0) * 0.6
    mid_weight = 0.1
    
    df['combined_volatility'] = (short_weight * df['short_std_vol_weighted_return'] +
                                 mid_weight * df['mid_std_vol_weighted_return'] +
                                 long_weight * df['long_std_vol_weighted_return'])
    
    # Incorporate Liquidity Measure
    df['liquidity'] = df['volume'].rolling(window=long_term_lookback).mean()
    liquidity_threshold = df['liquidity'].median()
    
    df['final_factor'] = (df['volume_weighted_return'] / df['combined_volatility']).where(df['liquidity'] >= liquidity_threshold, 
                                                                                           df['volume_weighted_return'] / (df['combined_volatility'] * 2))
    
    return df['final_factor'].dropna()
