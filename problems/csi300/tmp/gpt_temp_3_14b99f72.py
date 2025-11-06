import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate Daily True Range
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    data['TR'] = np.maximum(np.maximum(high_low, high_close_prev), low_close_prev)
    
    # Identify High Volatility Periods using rolling median
    window = 20
    data['TR_median'] = data['TR'].rolling(window=window, min_periods=1).median()
    data['high_vol_flag'] = (data['TR'] > data['TR_median']).astype(int)
    
    # Cluster consecutive high volatility days
    data['cluster_id'] = 0
    cluster_counter = 0
    in_cluster = False
    
    for i in range(len(data)):
        if data['high_vol_flag'].iloc[i] == 1:
            if not in_cluster:
                cluster_counter += 1
                in_cluster = True
            data.loc[data.index[i], 'cluster_id'] = cluster_counter
        else:
            in_cluster = False
    
    # Calculate cluster duration
    cluster_durations = data.groupby('cluster_id')['cluster_id'].transform('count')
    data['cluster_duration'] = np.where(data['cluster_id'] > 0, cluster_durations, 0)
    
    # Calculate intraday return
    data['intra_return'] = (data['close'] - data['open']) / data['open']
    
    # Calculate previous day return
    data['prev_return'] = (data['close'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    
    # Compute reversal magnitude
    data['reversal'] = -np.sign(data['prev_return']) * data['intra_return']
    data['reversal_strength'] = abs(data['reversal'])
    
    # Calculate volume metrics for clusters
    cluster_volumes = data[data['cluster_id'] > 0].groupby('cluster_id')['volume'].mean()
    data['cluster_avg_volume'] = data['cluster_id'].map(cluster_volumes).fillna(0)
    
    # Weight reversal by cluster duration with recent priority
    data['weighted_reversal'] = data['reversal_strength'] * data['cluster_duration']
    
    # Apply exponential decay for recent cluster priority
    decay_window = 5
    if len(data) > decay_window:
        weights = np.exp(-np.arange(decay_window) / decay_window)
        data['recency_weight'] = 0
        for i in range(len(data)):
            if i >= decay_window:
                recent_clusters = data['cluster_id'].iloc[i-decay_window:i]
                if recent_clusters.max() > 0:
                    data.loc[data.index[i], 'recency_weight'] = np.average(
                        recent_clusters > 0, weights=weights
                    )
    
    # Adjust by volume confirmation
    data['volume_ratio'] = data['volume'] / data['cluster_avg_volume'].replace(0, np.nan)
    data['volume_adjustment'] = np.where(
        data['volume_ratio'] > 1, 
        np.minimum(data['volume_ratio'], 2),  # Cap enhancement at 2x
        np.maximum(data['volume_ratio'], 0.5)  # Cap diminishment at 0.5x
    )
    
    # Final factor calculation
    data['factor'] = (
        data['weighted_reversal'] * 
        (1 + data['recency_weight']) * 
        data['volume_adjustment']
    )
    
    return data['factor']
