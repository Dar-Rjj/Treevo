import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 10-day price momentum
    momentum = df['close'] / df['close'].shift(10) - 1
    
    # Apply exponential decay with λ=0.94
    decay_factor = 0.94
    decayed_momentum = momentum.copy()
    for i in range(1, len(momentum)):
        if not pd.isna(decayed_momentum.iloc[i-1]):
            decayed_momentum.iloc[i] = decay_factor * decayed_momentum.iloc[i-1] + (1 - decay_factor) * momentum.iloc[i]
    
    # Calculate 20-day volume percentile
    volume_percentile = df['volume'].rolling(window=20, min_periods=1).apply(
        lambda x: (x[-1] > np.percentile(x, 80)) if len(x) >= 20 else False
    )
    
    # Identify high-volume clusters (consecutive days with volume > 80th percentile)
    volume_clusters = pd.Series(0, index=df.index)
    cluster_duration = pd.Series(0, index=df.index)
    
    current_cluster = 0
    current_duration = 0
    
    for i in range(len(volume_percentile)):
        if volume_percentile.iloc[i]:
            current_duration += 1
            if current_duration >= 2:
                current_cluster += 1
            volume_clusters.iloc[i] = current_cluster
            cluster_duration.iloc[i] = current_duration
        else:
            current_cluster = 0
            current_duration = 0
            volume_clusters.iloc[i] = 0
            cluster_duration.iloc[i] = 0
    
    # Calculate cluster intensity (normalized volume within cluster)
    cluster_intensity = pd.Series(0.0, index=df.index)
    for cluster_id in volume_clusters.unique():
        if cluster_id > 0:
            cluster_mask = volume_clusters == cluster_id
            cluster_volumes = df.loc[cluster_mask, 'volume']
            if len(cluster_volumes) > 0:
                max_volume = cluster_volumes.max()
                cluster_intensity.loc[cluster_mask] = cluster_volumes / max_volume
    
    # Apply momentum amplification based on volume clusters
    amplified_momentum = decayed_momentum.copy()
    cluster_mask = (volume_clusters > 0) & (cluster_duration >= 2)
    
    # Multiply momentum by cluster intensity and scale by cluster duration
    amplification_factor = 1 + (cluster_intensity * cluster_duration / 10)
    amplified_momentum.loc[cluster_mask] = decayed_momentum.loc[cluster_mask] * amplification_factor.loc[cluster_mask]
    
    return amplified_momentum
