import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required rolling statistics
    df['volume_ma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
    df['volume_ma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
    df['high_low_ma_5'] = (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    df['rolling_max_high_5'] = df['high'].rolling(window=5, min_periods=1).max()
    df['rolling_min_low_5'] = df['low'].rolling(window=5, min_periods=1).min()
    df['rolling_max_high_3'] = df['high'].rolling(window=3, min_periods=1).max()
    
    # Calculate volume quantile (70th percentile over 20 days)
    df['volume_quantile_70'] = df['volume'].rolling(window=20, min_periods=1).quantile(0.7)
    
    # Calculate autocorrelation for close prices (lag 3, window 5)
    def autocorr_3_5(x):
        if len(x) < 5:
            return 0
        return x.autocorr(lag=3) if not pd.isna(x.autocorr(lag=3)) else 0
    
    df['close_autocorr_3_5'] = df['close'].rolling(window=5, min_periods=1).apply(autocorr_3_5, raw=False)
    
    # Calculate count functions
    def count_close_gt_prev(x):
        return (x > x.shift(1)).sum()
    
    def count_volume_gt_prev(x):
        return (x > x.shift(1)).sum()
    
    df['count_close_up_5'] = df['close'].rolling(window=5, min_periods=1).apply(count_close_gt_prev, raw=False)
    df['count_volume_up_3'] = df['volume'].rolling(window=3, min_periods=1).apply(count_volume_gt_prev, raw=False)
    
    # Calculate price ratios with proper shifting
    df['close_ratio_3'] = df['close'] / df['close'].shift(3)
    df['close_ratio_5'] = df['close'] / df['close'].shift(5)
    df['close_ratio_8'] = df['close'] / df['close'].shift(8)
    df['close_ratio_10'] = df['close'] / df['close'].shift(10)
    
    # Replace infinite values and NaN with 0
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Calculate individual alpha components
    for i in range(len(df)):
        if i < 10:  # Need sufficient history
            alpha.iloc[i] = 0
            continue
            
        row = df.iloc[i]
        prev_row = df.iloc[i-1] if i > 0 else row
        
        # 1. Intraday Pressure with Volume Efficiency
        high_low_range = row['high'] - row['low']
        if high_low_range > 0:
            intraday_pressure_1 = ((row['close'] - row['low']) / high_low_range * 
                                 np.sign(row['volume'] - prev_row['volume']) * 
                                 row['volume'] / max(row['volume_ma_20'], 1e-6))
            
            intraday_pressure_2 = ((row['high'] - row['close']) / high_low_range * 
                                 np.sign(row['close'] - prev_row['close']) * 
                                 row['volume'] * abs(row['close'] - row['open']) / high_low_range)
        else:
            intraday_pressure_1 = intraday_pressure_2 = 0
        
        # 2. Volatility-Efficient Breakout with Order Flow
        breakout_1 = ((row['close'] - row['rolling_max_high_5']) * 
                     abs(row['close'] - row['open']) / max(high_low_range, 1e-6) * 
                     (2*row['close'] - row['high'] - row['low']) / max(high_low_range, 1e-6))
        
        breakout_2 = ((row['close'] - row['rolling_min_low_5']) * 
                     (row['close'] - row['open']) / max(high_low_range, 1e-6) * 
                     (2*row['close'] - row['high'] - row['low']) / max(high_low_range, 1e-6))
        
        # 3. Multi-Timeframe Momentum Acceleration
        momentum_1 = ((row['close_ratio_3'] - row['close_ratio_8']) * 
                     (1 if row['volume'] > row['volume_quantile_70'] else 0) * 
                     row['volume'] * (row['close'] - row['open']) / max(high_low_range, 1e-6))
        
        momentum_2 = ((row['close_ratio_5'] - row['close_ratio_10']) * 
                     row['close_autocorr_3_5'] * 
                     np.sign(row['volume'] - df.iloc[i-5]['volume'] if i >= 5 else 0))
        
        # 4. Volume-Weighted Trend with Afternoon Dynamics
        trend_1 = (((row['close'] - df.iloc[i-5]['close']) / max(df.iloc[i-5]['close'], 1e-6)) * 
                  row['volume'] / max(row['volume_ma_20'], 1e-6) * 
                  (row['close'] - (row['open'] + row['high'])/2) / max((row['open'] + row['high'])/2, 1e-6))
        
        trend_2 = (((row['close'] - df.iloc[i-10]['close']) / max(df.iloc[i-10]['close'], 1e-6)) * 
                  row['volume'] * abs(row['close'] - row['open']) / max(high_low_range, 1e-6) * 
                  (row['volume'] / max(prev_row['volume'], 1e-6)))
        
        # 5. Order Flow Confirmed Range Expansion
        range_exp_1 = ((high_low_range / max(row['high_low_ma_5'], 1e-6)) * 
                      (2*row['close'] - row['high'] - row['low']) / max(high_low_range, 1e-6) * 
                      row['volume'] / max(row['volume_ma_10'], 1e-6))
        
        range_exp_2 = ((row['close'] - row['rolling_max_high_3']) * 
                      high_low_range / max(row['high_low_ma_5'], 1e-6) * 
                      (2*row['close'] - row['high'] - row['low']) / max(high_low_range, 1e-6))
        
        # 6. Momentum Persistence with Volume Divergence
        persistence_1 = (row['count_close_up_5'] * 
                        (row['close_ratio_5'] - (df.iloc[i-3]['close'] / df.iloc[i-8]['close'] if i >= 8 else 0)) * 
                        np.sign(row['volume'] - df.iloc[i-5]['volume'] if i >= 5 else 0))
        
        persistence_2 = (row['close_ratio_3'] * 
                        row['count_volume_up_3'] * 
                        (row['close'] - row['open']) / max(high_low_range, 1e-6))
        
        # Combine all components with equal weighting
        alpha.iloc[i] = (intraday_pressure_1 + intraday_pressure_2 + 
                        breakout_1 + breakout_2 + 
                        momentum_1 + momentum_2 + 
                        trend_1 + trend_2 + 
                        range_exp_1 + range_exp_2 + 
                        persistence_1 + persistence_2)
    
    # Final cleaning
    alpha = alpha.replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha
