import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence factor that detects multi-scale patterns 
    in price and volume to identify divergences that predict future returns.
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-scale fractal analysis for price
    def calculate_fractal_dimension(series, window=20):
        """Calculate fractal dimension using Hurst exponent approximation"""
        lags = range(2, min(15, window//2))
        tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Price fractal dimensions at different scales
    data['price_fractal_short'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: calculate_fractal_dimension(x), raw=False
    )
    data['price_fractal_medium'] = data['close'].rolling(window=30, min_periods=15).apply(
        lambda x: calculate_fractal_dimension(x), raw=False
    )
    data['price_fractal_long'] = data['close'].rolling(window=60, min_periods=30).apply(
        lambda x: calculate_fractal_dimension(x), raw=False
    )
    
    # 2. Volume fractal analysis
    def calculate_volume_clustering(volume_series, window=20):
        """Measure volume clustering patterns"""
        volume_std = volume_series.rolling(window=window, min_periods=10).std()
        volume_mean = volume_series.rolling(window=window, min_periods=10).mean()
        return (volume_std / (volume_mean + 1e-8)).fillna(0)
    
    data['volume_clustering_short'] = calculate_volume_clustering(data['volume'], 10)
    data['volume_clustering_medium'] = calculate_volume_clustering(data['volume'], 30)
    data['volume_clustering_long'] = calculate_volume_clustering(data['volume'], 60)
    
    # Volume fractal dimension
    data['volume_fractal'] = data['volume'].rolling(window=30, min_periods=15).apply(
        lambda x: calculate_fractal_dimension(x), raw=False
    )
    
    # 3. Local minima/maxima detection for pattern completeness
    def find_local_extrema(price_series, window=5):
        """Find local minima and maxima"""
        local_max = (price_series == price_series.rolling(window=window, center=True).max())
        local_min = (price_series == price_series.rolling(window=window, center=True).min())
        return local_max.astype(int) - local_min.astype(int)
    
    data['price_extrema'] = find_local_extrema(data['close'], 5)
    
    # Pattern symmetry measurement
    def calculate_pattern_symmetry(price_series, window=20):
        """Measure pattern symmetry using autocorrelation"""
        returns = price_series.pct_change().fillna(0)
        autocorr = returns.rolling(window=window, min_periods=10).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
        return autocorr.abs()
    
    data['pattern_symmetry'] = calculate_pattern_symmetry(data['close'], 20)
    
    # 4. Price-Volume fractal divergence calculation
    def calculate_fractal_divergence(row):
        """Calculate divergence between price and volume fractal patterns"""
        # Short-term divergence
        price_short = row.get('price_fractal_short', 0)
        vol_cluster_short = row.get('volume_clustering_short', 0)
        
        # Medium-term divergence
        price_medium = row.get('price_fractal_medium', 0)
        vol_cluster_medium = row.get('volume_clustering_medium', 0)
        
        # Long-term divergence
        price_long = row.get('price_fractal_long', 0)
        vol_cluster_long = row.get('volume_clustering_long', 0)
        
        # Calculate weighted divergence score
        short_div = (price_short - vol_cluster_short) if not np.isnan(price_short) and not np.isnan(vol_cluster_short) else 0
        medium_div = (price_medium - vol_cluster_medium) if not np.isnan(price_medium) and not np.isnan(vol_cluster_medium) else 0
        long_div = (price_long - vol_cluster_long) if not np.isnan(price_long) and not np.isnan(vol_cluster_long) else 0
        
        # Weight by timeframe consistency
        weights = [0.4, 0.35, 0.25]  # More weight to shorter timeframes
        divergences = [short_div, medium_div, long_div]
        
        valid_divs = [d for d in divergences if not np.isnan(d)]
        valid_weights = weights[:len(valid_divs)]
        
        if len(valid_divs) > 0:
            # Normalize weights
            valid_weights = [w/sum(valid_weights) for w in valid_weights]
            return sum(d * w for d, w in zip(valid_divs, valid_weights))
        else:
            return 0
    
    # Calculate divergence factor
    factor_values = []
    for idx, row in data.iterrows():
        divergence = calculate_fractal_divergence(row)
        factor_values.append(divergence)
    
    factor_series = pd.Series(factor_values, index=data.index)
    
    # 5. Final normalization and smoothing
    factor_series = factor_series.rolling(window=5, min_periods=3).mean()
    factor_series = (factor_series - factor_series.rolling(window=60, min_periods=30).mean()) / (
        factor_series.rolling(window=60, min_periods=30).std() + 1e-8
    )
    
    return factor_series.fillna(0)
