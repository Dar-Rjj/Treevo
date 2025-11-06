import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from scipy.optimize import curve_fit

def heuristics_v2(df):
    """
    Price-Volume Fractal Dimension Divergence factor combining multi-scale roughness analysis
    with volume distribution asymmetry and nonlinear dependency measures.
    """
    df = df.copy()
    
    # 1. Multi-Scale Price Roughness
    def hurst_exponent(series, max_lag=20):
        """Calculate Hurst exponent using R/S analysis"""
        lags = range(2, max_lag+1)
        tau = []
        for lag in lags:
            rs_values = []
            for i in range(0, len(series)-lag, lag):
                sub_series = series[i:i+lag]
                if len(sub_series) < 2:
                    continue
                mean_val = np.mean(sub_series)
                deviations = sub_series - mean_val
                z = np.cumsum(deviations)
                r = np.max(z) - np.min(z)
                s = np.std(sub_series)
                if s > 0:
                    rs_values.append(r/s)
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
            else:
                tau.append(np.nan)
        
        if len(tau) > 2:
            lags_log = np.log(lags[:len(tau)])
            valid_idx = ~np.isnan(tau)
            if np.sum(valid_idx) > 2:
                try:
                    hurst, _ = np.polyfit(lags_log[valid_idx], tau[valid_idx], 1)
                    return hurst
                except:
                    return np.nan
        return np.nan
    
    def fractal_dimension_box_counting(high, low, periods=20):
        """Calculate fractal dimension using box-counting method"""
        dimensions = []
        for i in range(periods, len(high)):
            window_high = high[i-periods:i]
            window_low = low[i-periods:i]
            
            if len(window_high) < periods:
                dimensions.append(np.nan)
                continue
            
            price_range = window_high - window_low
            max_range = np.max(price_range)
            min_range = np.min(price_range)
            
            if max_range <= min_range or max_range == 0:
                dimensions.append(np.nan)
                continue
            
            # Simple box counting approximation
            num_boxes = len(set(np.floor((price_range - min_range) / (max_range - min_range) * 10).astype(int)))
            if num_boxes > 0:
                dim = np.log(num_boxes) / np.log(periods)
                dimensions.append(dim)
            else:
                dimensions.append(np.nan)
        
        return pd.Series(dimensions, index=high.index[periods:]).reindex(high.index)
    
    # Calculate price roughness measures
    df['hurst_5d'] = df['close'].rolling(30).apply(lambda x: hurst_exponent(x.dropna()), raw=False)
    df['hurst_20d'] = df['close'].rolling(60).apply(lambda x: hurst_exponent(x.dropna()), raw=False)
    df['fractal_dim'] = fractal_dimension_box_counting(df['high'], df['low'])
    
    # 2. Volume Distribution Asymmetry
    def volume_entropy(volume_series, bins=10):
        """Calculate Shannon entropy of volume distribution"""
        hist, _ = np.histogram(volume_series.dropna(), bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) > 1:
            return entropy(hist)
        return np.nan
    
    df['volume_skew_5d'] = df['volume'].rolling(5).apply(skew, raw=True)
    df['volume_skew_20d'] = df['volume'].rolling(20).apply(skew, raw=True)
    df['volume_entropy_10d'] = df['volume'].rolling(10).apply(volume_entropy, raw=False)
    
    # Volume clustering measure
    def volume_clustering(volume_series):
        """Measure volume clustering using spatial statistics"""
        if len(volume_series) < 5:
            return np.nan
        vol_diff = volume_series.diff().dropna()
        if len(vol_diff) < 4:
            return np.nan
        # Simple clustering measure based on consecutive same-sign changes
        sign_changes = np.sign(vol_diff)
        cluster_strength = (sign_changes.rolling(3).apply(
            lambda x: np.sum(x == x.iloc[0]) if len(x) == 3 else np.nan, raw=False
        )).mean()
        return cluster_strength
    
    df['volume_cluster_10d'] = df['volume'].rolling(10).apply(volume_clustering, raw=False)
    
    # 3. Fractal Dimension Divergence
    def kl_divergence(p, q):
        """Calculate Kullback-Leibler divergence between two distributions"""
        p = np.array(p)
        q = np.array(q)
        p = p[p > 0]
        q = q[q > 0]
        if len(p) == 0 or len(q) == 0:
            return np.nan
        return np.sum(p * np.log(p / q))
    
    # Calculate divergence between price and volume fractal patterns
    df['price_vol_divergence'] = np.nan
    for i in range(20, len(df)):
        price_window = df['close'].iloc[i-20:i]
        volume_window = df['volume'].iloc[i-20:i]
        
        if len(price_window.dropna()) < 10 or len(volume_window.dropna()) < 10:
            continue
            
        # Normalize and create histograms
        price_norm = (price_window - price_window.min()) / (price_window.max() - price_window.min() + 1e-8)
        volume_norm = (volume_window - volume_window.min()) / (volume_window.max() - volume_window.min() + 1e-8)
        
        price_hist, _ = np.histogram(price_norm, bins=5, density=True)
        volume_hist, _ = np.histogram(volume_norm, bins=5, density=True)
        
        price_hist = price_hist[price_hist > 0]
        volume_hist = volume_hist[volume_hist > 0]
        
        if len(price_hist) > 1 and len(volume_hist) > 1:
            divergence = kl_divergence(price_hist, volume_hist)
            df.iloc[i, df.columns.get_loc('price_vol_divergence')] = divergence
    
    # 4. Nonlinear Dependency Measures
    def mutual_information_estimate(x, y, bins=5):
        """Estimate mutual information between two series"""
        x_clean = x.dropna()
        y_clean = y.dropna()
        
        if len(x_clean) < 10 or len(y_clean) < 10:
            return np.nan
            
        # Create 2D histogram
        hist_2d, _, _ = np.histogram2d(x_clean, y_clean, bins=bins, density=True)
        hist_x, _ = np.histogram(x_clean, bins=bins, density=True)
        hist_y, _ = np.histogram(y_clean, bins=bins, density=True)
        
        # Calculate mutual information
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if hist_2d[i,j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                    mi += hist_2d[i,j] * np.log(hist_2d[i,j] / (hist_x[i] * hist_y[j]))
        
        return mi
    
    df['mutual_info_10d'] = df['close'].rolling(10).apply(
        lambda x: mutual_information_estimate(x, df['volume'].loc[x.index]), raw=False
    )
    
    # 5. Composite Factor Generation
    # Normalize components
    components = ['hurst_5d', 'hurst_20d', 'fractal_dim', 'volume_skew_5d', 
                 'volume_skew_20d', 'volume_entropy_10d', 'volume_cluster_10d', 
                 'price_vol_divergence', 'mutual_info_10d']
    
    # Calculate z-scores for each component
    z_scores = {}
    for comp in components:
        if comp in df.columns:
            mean_val = df[comp].rolling(60).mean()
            std_val = df[comp].rolling(60).std()
            z_scores[comp] = (df[comp] - mean_val) / (std_val + 1e-8)
    
    # Combine components with regime-dependent weights
    # Higher weight to divergence and mutual information during normal periods
    volatility_regime = df['close'].pct_change().rolling(20).std()
    low_vol = volatility_regime < volatility_regime.rolling(60).quantile(0.3)
    high_vol = volatility_regime > volatility_regime.rolling(60).quantile(0.7)
    
    # Base weights
    weights = {
        'price_vol_divergence': 0.25,
        'mutual_info_10d': 0.20,
        'fractal_dim': 0.15,
        'hurst_5d': 0.10,
        'volume_skew_5d': 0.10,
        'volume_entropy_10d': 0.10,
        'volume_cluster_10d': 0.10
    }
    
    # Adjust weights based on volatility regime
    factor = pd.Series(index=df.index, dtype=float)
    for i in range(60, len(df)):
        if i not in z_scores['price_vol_divergence'].index:
            continue
            
        current_weights = weights.copy()
        
        # High volatility: emphasize fractal dimension and volume clustering
        if high_vol.iloc[i]:
            current_weights['fractal_dim'] *= 1.5
            current_weights['volume_cluster_10d'] *= 1.5
            current_weights['price_vol_divergence'] *= 0.7
        
        # Low volatility: emphasize divergence and mutual information
        if low_vol.iloc[i]:
            current_weights['price_vol_divergence'] *= 1.5
            current_weights['mutual_info_10d'] *= 1.5
        
        # Calculate weighted sum
        weighted_sum = 0
        total_weight = 0
        for comp, weight in current_weights.items():
            if comp in z_scores and not pd.isna(z_scores[comp].iloc[i]):
                weighted_sum += z_scores[comp].iloc[i] * weight
                total_weight += weight
        
        if total_weight > 0:
            factor.iloc[i] = weighted_sum / total_weight
    
    # Final scaling by recent signal effectiveness
    signal_persistence = factor.rolling(5).std()
    factor_scaled = factor / (signal_persistence + 1e-8)
    
    return factor_scaled
