import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] / df['open']) - 1
    
    # Enhance with Volume-Adjusted High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_adjusted_high_low_diff'] = df['volume'] / df['high_low_diff']
    
    # Incorporate Trading Activity Intensity
    df['average_volume_20d'] = df['volume'].rolling(window=20).mean()
    df['volume_to_average_ratio'] = df['volume'] / df['average_volume_20d']
    
    # Calculate Adjusted Intraday Reversal
    df['intraday_range'] = df['high'] - df['low']
    df['adjusted_intraday_reversal'] = df['intraday_range'] * df['volume']
    
    # Calculate 14-Day Volume-Weighted Price Change
    df['daily_return'] = df['close'].pct_change()
    df['volume_weighted_price_change_14d'] = (df['daily_return'] * df['volume']).rolling(window=14).sum() / df['volume'].rolling(window=14).sum()
    
    # Combine Adjusted Intraday Reversal and 14-Day Volume-Weighted Price Change
    df['combined_factor'] = df['adjusted_intraday_reversal'] * df['volume_weighted_price_change_14d']
    
    # Consider Consecutive Up/Down Days and Extreme Movement
    df['up_day'] = (df['close'] > df['open']).astype(int)
    df['down_day'] = (df['open'] > df['close']).astype(int)
    df['consecutive_up_days'] = df['up_day'].groupby((df['up_day'] != df['up_day'].shift()).cumsum()).cumcount()
    df['consecutive_down_days'] = df['down_day'].groupby((df['down_day'] != df['down_day'].shift()).cumsum()).cumcount()
    df['extreme_movement'] = df['high_low_diff'].apply(lambda x: 1 if x > df['high_low_diff'].quantile(0.9) else 0)
    df['combined_factor'] *= (1 + df['extreme_movement'])
    
    # Incorporate Volume Influence
    df['combined_intraday_factor'] = df['intraday_return'] * df['volume_adjusted_high_low_diff']
    df['weighted_combined_factor'] = df['combined_intraday_factor'] * df['volume_to_average_ratio']
    
    # Calculate Daily Log Returns
    df['log_return'] = df['close'].apply(np.log) - df['close'].shift(1).apply(np.log)
    
    # Compute Daily Volatility
    df['daily_volatility_20d'] = df['log_return'].rolling(window=20).std()
    
    # Final Alpha Factor
    df['final_alpha_factor'] = (df['weighted_combined_factor'] + 
                                df['volume_weighted_price_change_14d'] + 
                                df['adjusted_intraday_reversal'] + 
                                df['daily_volatility_20d'])
    df['final_alpha_factor'] /= df['volume']
    
    return df['final_alpha_factor'].dropna()
