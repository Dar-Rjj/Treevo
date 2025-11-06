import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Price Efficiency with Volume Clustering
    # Calculate Intraday Efficiency Ratio
    prev_close = data['close'].shift(1)
    high_low_range = data['high'] - data['low']
    high_prev_range = abs(data['high'] - prev_close)
    low_prev_range = abs(data['low'] - prev_close)
    
    max_range = pd.concat([high_low_range, high_prev_range, low_prev_range], axis=1).max(axis=1)
    efficiency_ratio = abs((data['close'] - data['open']) / np.maximum(max_range, 1e-8))
    
    # Analyze Volume Clusters
    volume_mean = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_std = data['volume'].rolling(window=5, min_periods=1).std()
    volume_zscore = (data['volume'] - volume_mean) / np.maximum(volume_std, 1e-8)
    
    # Identify volume clusters
    high_volume_cluster = (volume_zscore > 2).astype(int)
    low_volume_cluster = (volume_zscore < -1).astype(int)
    
    # Calculate cluster duration
    cluster_duration = pd.Series(0, index=data.index)
    current_duration = 0
    for i in range(len(data)):
        if high_volume_cluster.iloc[i] or low_volume_cluster.iloc[i]:
            current_duration += 1
        else:
            current_duration = 0
        cluster_duration.iloc[i] = current_duration
    
    # Combine Efficiency with Volume
    base_factor = efficiency_ratio * volume_zscore
    adjusted_factor = base_factor / np.maximum(np.sqrt(cluster_duration), 1)
    
    # Price-Volume Divergence Across Timeframes
    # Compute price trends (slopes)
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            y = series.iloc[i-window+1:i+1].values
            if len(y) == window:
                slope = (window * np.sum(x*y) - np.sum(x) * np.sum(y)) / (window * np.sum(x*x) - np.sum(x)**2)
                slopes.iloc[i] = slope
        return slopes.fillna(0)
    
    price_trend_4 = calculate_slope(data['close'], 4)
    price_trend_3 = calculate_slope(data['close'], 3)
    price_trend_10 = calculate_slope(data['close'], 10)
    
    # Calculate volume-price correlations
    def rolling_correlation(price_series, volume_series, window):
        corr = price_series.rolling(window=window).corr(volume_series)
        return corr.fillna(0)
    
    corr_4 = rolling_correlation(data['close'], data['volume'], 4)
    corr_3 = rolling_correlation(data['close'], data['volume'], 3)
    corr_10 = rolling_correlation(data['close'], data['volume'], 10)
    
    # Identify divergences
    divergence_4 = ((price_trend_4 > 0) & (corr_4 < 0)) | ((price_trend_4 < 0) & (corr_4 > 0))
    divergence_3 = ((price_trend_3 > 0) & (corr_3 < 0)) | ((price_trend_3 < 0) & (corr_3 > 0))
    divergence_10 = ((price_trend_10 > 0) & (corr_10 < 0)) | ((price_trend_10 < 0) & (corr_10 > 0))
    
    divergence_score = (divergence_4.astype(int) + divergence_3.astype(int) + divergence_10.astype(int)) / 3
    
    # Order Flow Efficiency at Key Levels
    # Calculate order flow efficiency
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    order_flow_efficiency = data['amount'] / np.maximum(data['volume'] * typical_price, 1e-8)
    
    # Identify support/resistance levels
    recent_high = data['high'].rolling(window=10, min_periods=1).max()
    recent_low = data['low'].rolling(window=10, min_periods=1).min()
    
    # Calculate level proximity
    high_proximity = (data['close'] - recent_low) / np.maximum(recent_high - recent_low, 1e-8)
    level_proximity = 1 - abs(high_proximity - 0.5) * 2  # Closer to extremes = higher score
    
    flow_factor = order_flow_efficiency * level_proximity
    
    # Volatility Compression Expansion Signal
    # Measure range compression
    current_range = data['high'] - data['low']
    avg_range_5d = current_range.rolling(window=5, min_periods=1).mean()
    range_compression = current_range / np.maximum(avg_range_5d, 1e-8)
    
    # Volume Z-score change
    volume_zscore_change = volume_zscore.diff().fillna(0)
    
    compression_factor = -range_compression * volume_zscore_change
    
    # Momentum Regime Transition Timing
    # Calculate momentum acceleration (2nd derivative)
    momentum_1d = data['close'].pct_change(1).fillna(0)
    momentum_acceleration = momentum_1d.diff().fillna(0)
    
    # Analyze momentum duration
    momentum_sign = np.sign(momentum_1d)
    momentum_duration = pd.Series(0, index=data.index)
    current_run = 0
    for i in range(1, len(data)):
        if momentum_sign.iloc[i] == momentum_sign.iloc[i-1] and momentum_sign.iloc[i] != 0:
            current_run += 1
        else:
            current_run = 1
        momentum_duration.iloc[i] = current_run
    
    # Historical momentum pattern analysis
    avg_momentum_duration = momentum_duration.rolling(window=20, min_periods=1).mean()
    duration_ratio = momentum_duration / np.maximum(avg_momentum_duration, 1)
    
    # Transition probability (higher when momentum is extended)
    transition_probability = 1 / np.maximum(1 + np.exp(-(duration_ratio - 2)), 1e-8)
    
    # Volume confirmation
    volume_confirmation = np.sign(momentum_1d) * volume_zscore
    
    momentum_factor = transition_probability * volume_confirmation
    
    # Combine all factors with equal weights
    final_factor = (
        adjusted_factor.fillna(0) + 
        divergence_score.fillna(0) + 
        flow_factor.fillna(0) + 
        compression_factor.fillna(0) + 
        momentum_factor.fillna(0)
    ) / 5
    
    return final_factor
