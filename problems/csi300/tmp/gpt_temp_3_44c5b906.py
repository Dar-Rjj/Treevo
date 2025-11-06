import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence with Fractal Market Analysis
    Combines fractal price patterns and volume structure to detect divergences
    """
    data = df.copy()
    
    # Helper function to detect local extrema
    def detect_swing_highs(series, window=5):
        highs = []
        for i in range(window, len(series) - window):
            if all(series[i] > series[i-j] for j in range(1, window+1)) and \
               all(series[i] > series[i+j] for j in range(1, window+1)):
                highs.append(i)
        return highs
    
    def detect_swing_lows(series, window=5):
        lows = []
        for i in range(window, len(series) - window):
            if all(series[i] < series[i-j] for j in range(1, window+1)) and \
               all(series[i] < series[i+j] for j in range(1, window+1)):
                lows.append(i)
        return lows
    
    # Calculate fractal dimension using box-counting method
    def fractal_dimension(price_series, window=20, box_sizes=None):
        if box_sizes is None:
            box_sizes = [2, 4, 8, 16]
        
        dimensions = []
        for i in range(window, len(price_series)):
            window_data = price_series[i-window:i]
            if len(window_data) < window:
                dimensions.append(np.nan)
                continue
            
            price_range = window_data.max() - window_data.min()
            if price_range == 0:
                dimensions.append(0)
                continue
            
            counts = []
            for box_size in box_sizes:
                if box_size >= len(window_data):
                    continue
                num_boxes = np.ceil(price_range / (window_data.std() / box_size))
                if num_boxes == 0:
                    continue
                counts.append(np.log(num_boxes))
            
            if len(counts) < 2:
                dimensions.append(np.nan)
                continue
            
            # Linear regression to estimate fractal dimension
            x = np.log([bs for bs in box_sizes[:len(counts)]])
            y = np.array(counts)
            slope = np.polyfit(x, y, 1)[0]
            dimensions.append(slope)
        
        return [np.nan] * window + dimensions
    
    # Volume clustering analysis
    def volume_clustering(volume_series, window=10):
        clustering_scores = []
        for i in range(window, len(volume_series)):
            window_data = volume_series[i-window:i]
            volume_mean = window_data.mean()
            volume_std = window_data.std()
            
            if volume_std == 0:
                clustering_scores.append(0)
                continue
            
            # Count peaks and voids
            peaks = sum(1 for v in window_data if v > volume_mean + volume_std)
            voids = sum(1 for v in window_data if v < volume_mean - volume_std)
            
            clustering_score = (peaks - voids) / window
            clustering_scores.append(clustering_score)
        
        return [np.nan] * window + clustering_scores
    
    # Volume persistence (simplified Hurst exponent)
    def volume_persistence(volume_series, window=20):
        persistence_scores = []
        for i in range(window, len(volume_series)):
            window_data = volume_series[i-window:i]
            
            # Count consecutive trend days
            trends = []
            current_trend = 0
            for j in range(1, len(window_data)):
                if window_data[j] > window_data[j-1]:
                    if current_trend > 0:
                        current_trend += 1
                    else:
                        trends.append(current_trend)
                        current_trend = 1
                elif window_data[j] < window_data[j-1]:
                    if current_trend < 0:
                        current_trend -= 1
                    else:
                        trends.append(current_trend)
                        current_trend = -1
                else:
                    trends.append(current_trend)
                    current_trend = 0
            
            if trends:
                avg_persistence = np.mean(np.abs(trends))
            else:
                avg_persistence = 0
            
            persistence_scores.append(avg_persistence)
        
        return [np.nan] * window + persistence_scores
    
    # Calculate all components
    close_prices = data['close'].values
    volumes = data['volume'].values
    
    # Price fractal dimensions
    short_fractal = fractal_dimension(close_prices, window=5)
    long_fractal = fractal_dimension(close_prices, window=20)
    
    # Volume analysis
    vol_clustering = volume_clustering(volumes, window=10)
    vol_persistence = volume_persistence(volumes, window=20)
    
    # Multi-timeframe divergence
    fractal_divergence = []
    price_volume_correlation = []
    
    for i in range(20, len(close_prices)):
        if i >= len(short_fractal) or i >= len(long_fractal) or \
           i >= len(vol_clustering) or i >= len(vol_persistence):
            fractal_divergence.append(np.nan)
            price_volume_correlation.append(np.nan)
            continue
        
        # Fractal dimension divergence
        if not np.isnan(short_fractal[i]) and not np.isnan(long_fractal[i]):
            divergence = short_fractal[i] - long_fractal[i]
        else:
            divergence = 0
        
        # Price-volume fractal correlation (simplified)
        price_window = close_prices[i-10:i]
        vol_window = volumes[i-10:i]
        
        if len(price_window) == 10 and len(vol_window) == 10:
            price_changes = np.diff(price_window)
            vol_changes = np.diff(vol_window)
            
            if len(price_changes) > 1 and len(vol_changes) > 1:
                correlation = np.corrcoef(price_changes, vol_changes)[0,1]
                if np.isnan(correlation):
                    correlation = 0
            else:
                correlation = 0
        else:
            correlation = 0
        
        fractal_divergence.append(divergence)
        price_volume_correlation.append(correlation)
    
    # Generate final signal
    signal = []
    for i in range(len(close_prices)):
        if i < 20:
            signal.append(0)
            continue
        
        idx = i - 20
        
        if idx >= len(fractal_divergence) or idx >= len(price_volume_correlation):
            signal.append(0)
            continue
        
        divergence = fractal_divergence[idx] if not np.isnan(fractal_divergence[idx]) else 0
        correlation = price_volume_correlation[idx] if not np.isnan(price_volume_correlation[idx]) else 0
        clustering = vol_clustering[i] if not np.isnan(vol_clustering[i]) else 0
        persistence = vol_persistence[i] if not np.isnan(vol_persistence[i]) else 0
        
        # Combined signal with regime weighting
        regime_weight = 1.0 + abs(persistence) * 0.1  # Higher weight for persistent regimes
        divergence_score = divergence * (1 - abs(correlation)) * clustering * regime_weight
        
        signal.append(divergence_score)
    
    return pd.Series(signal, index=data.index)
