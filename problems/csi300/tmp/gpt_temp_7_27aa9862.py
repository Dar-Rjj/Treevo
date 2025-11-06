import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate intraday range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate range changes for momentum acceleration
    df['range_3d_change'] = df['intraday_range'] / df['intraday_range'].shift(3) - 1
    df['range_6d_change'] = df['intraday_range'] / df['intraday_range'].shift(6) - 1
    
    # Calculate momentum acceleration
    df['momentum_acceleration'] = (df['range_3d_change'] - df['range_6d_change']) / (np.abs(df['range_6d_change']) + 1e-8)
    
    # Volume persistence analysis
    # Calculate 5-day volume correlation with time
    window_size = 5
    volume_corr = []
    volume_slope = []
    
    for i in range(len(df)):
        if i < window_size:
            volume_corr.append(np.nan)
            volume_slope.append(np.nan)
            continue
            
        window_data = df['volume'].iloc[i-window_size+1:i+1]
        time_index = np.arange(len(window_data))
        
        if window_data.std() == 0:
            corr = 0
            slope = 0
        else:
            corr = np.corrcoef(time_index, window_data)[0, 1]
            if np.isnan(corr):
                corr = 0
            # Calculate slope using linear regression
            slope = np.cov(time_index, window_data)[0, 1] / np.var(time_index)
            if np.isnan(slope):
                slope = 0
        
        volume_corr.append(corr)
        volume_slope.append(slope)
    
    df['volume_correlation'] = volume_corr
    df['volume_slope'] = volume_slope
    
    # Volume stability assessment
    df['volume_std_5d'] = df['volume'].rolling(window=5, min_periods=3).std()
    df['volume_avg_10d'] = df['volume'].rolling(window=10, min_periods=5).mean()
    df['volume_volatility_ratio'] = df['volume_std_5d'] / (df['volume_avg_10d'] + 1e-8)
    
    # Calculate volume persistence score
    df['volume_trend_strength'] = np.abs(df['volume_correlation']) * np.sign(df['volume_slope'])
    df['volume_stability_score'] = 1 / (1 + df['volume_volatility_ratio'])
    df['volume_persistence'] = df['volume_trend_strength'] * df['volume_stability_score']
    
    # Combine momentum acceleration with volume persistence
    df['factor'] = df['momentum_acceleration'] * df['volume_persistence']
    
    # Clean up intermediate columns
    result = df['factor'].copy()
    
    return result
