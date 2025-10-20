import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Volume-Price Efficiency with Fractal Market Dynamics alpha factor
    Combines efficiency metrics with fractal structure analysis across multiple time scales
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required columns
    cols_required = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in cols_required:
        if col not in df.columns:
            df[col] = 0.0
    
    # Calculate returns and log prices
    df['returns'] = df['close'].pct_change()
    df['log_close'] = np.log(df['close'])
    
    # Rolling windows for multi-scale analysis
    windows = [5, 13, 21]
    
    for i in range(max(windows), len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-scale efficiency analysis
        scale_efficiencies = []
        scale_weights = []
        
        for window in windows:
            if i >= window:
                window_data = current_data.iloc[-window:]
                
                # Price efficiency indicators
                price_efficiency = calculate_price_efficiency(window_data, window)
                volume_efficiency = calculate_volume_efficiency(window_data, window)
                
                # Fractal structure analysis
                fractal_metrics = calculate_fractal_metrics(window_data, window)
                
                # Scale-dependent efficiency score
                scale_score = price_efficiency * fractal_metrics['hurst_price'] - volume_efficiency * fractal_metrics['hurst_volume']
                scale_efficiencies.append(scale_score)
                scale_weights.append(fractal_metrics['persistence_strength'])
        
        if scale_efficiencies:
            # Weight efficiencies by persistence strength
            if sum(scale_weights) > 0:
                weighted_efficiency = np.average(scale_efficiencies, weights=scale_weights)
            else:
                weighted_efficiency = np.mean(scale_efficiencies)
            
            # Apply fractal regime adjustment
            regime_adjustment = calculate_regime_adjustment(current_data, windows)
            final_factor = weighted_efficiency * regime_adjustment
            
            result.iloc[i] = final_factor
        else:
            result.iloc[i] = 0.0
    
    # Fill initial NaN values
    result = result.fillna(0.0)
    
    return result

def calculate_price_efficiency(data, window):
    """Calculate price efficiency using Hurst exponent and path complexity"""
    
    if len(data) < window:
        return 0.0
    
    prices = data['close'].values
    
    # Simple Hurst exponent estimation using R/S analysis
    returns = np.diff(np.log(prices))
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate rescaled range
    mean_return = np.mean(returns)
    deviations = returns - mean_return
    cumulative_deviations = np.cumsum(deviations)
    range_val = np.max(cumulative_deviations) - np.min(cumulative_deviations)
    std_val = np.std(returns)
    
    if std_val > 0:
        hurst = np.log(range_val / std_val) / np.log(len(returns))
    else:
        hurst = 0.5
    
    # Path complexity (fractal dimension approximation)
    price_changes = np.diff(prices)
    total_movement = np.sum(np.abs(price_changes))
    net_movement = np.abs(prices[-1] - prices[0])
    
    if net_movement > 0:
        path_complexity = total_movement / net_movement
        fractal_dim = 2 - np.log(path_complexity) / np.log(len(price_changes)) if path_complexity > 1 else 1.0
    else:
        fractal_dim = 1.5
    
    # Efficiency score combining Hurst and fractal dimension
    efficiency = (1.0 - abs(hurst - 0.5)) * (2.0 - fractal_dim)
    
    return np.clip(efficiency, -1.0, 1.0)

def calculate_volume_efficiency(data, window):
    """Calculate volume efficiency using clustering and entropy measures"""
    
    if len(data) < window:
        return 0.0
    
    volumes = data['volume'].values
    
    if np.std(volumes) == 0:
        return 0.0
    
    # Volume clustering persistence
    volume_changes = np.diff(volumes)
    positive_clusters = count_clusters(volume_changes > 0)
    negative_clusters = count_clusters(volume_changes < 0)
    
    total_clusters = positive_clusters + negative_clusters
    if total_clusters > 0:
        clustering_persistence = max(positive_clusters, negative_clusters) / total_clusters
    else:
        clustering_persistence = 0.5
    
    # Volume entropy (simplified)
    volume_bins = np.histogram(volumes, bins=min(5, len(volumes)))[0]
    volume_probs = volume_bins / len(volumes)
    volume_probs = volume_probs[volume_probs > 0]
    
    if len(volume_probs) > 0:
        entropy = -np.sum(volume_probs * np.log(volume_probs))
        max_entropy = np.log(len(volume_probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        normalized_entropy = 0.0
    
    # Volume efficiency score
    volume_efficiency = (1.0 - clustering_persistence) * normalized_entropy
    
    return np.clip(volume_efficiency, 0.0, 1.0)

def calculate_fractal_metrics(data, window):
    """Calculate fractal metrics including Hurst exponents and persistence"""
    
    prices = data['close'].values
    volumes = data['volume'].values
    
    metrics = {
        'hurst_price': 0.5,
        'hurst_volume': 0.5,
        'persistence_strength': 0.0
    }
    
    if len(prices) < 10:
        return metrics
    
    # Price Hurst estimation
    price_returns = np.diff(np.log(prices))
    if len(price_returns) > 1 and np.std(price_returns) > 0:
        metrics['hurst_price'] = estimate_hurst_simple(price_returns)
    
    # Volume Hurst estimation
    if len(volumes) > 1 and np.std(volumes) > 0:
        volume_returns = np.diff(np.log(volumes + 1e-8))
        if np.std(volume_returns) > 0:
            metrics['hurst_volume'] = estimate_hurst_simple(volume_returns)
    
    # Persistence strength (absolute deviation from random walk)
    metrics['persistence_strength'] = abs(metrics['hurst_price'] - 0.5) + abs(metrics['hurst_volume'] - 0.5)
    
    return metrics

def calculate_regime_adjustment(data, windows):
    """Calculate fractal regime adjustment based on multi-scale analysis"""
    
    regime_scores = []
    
    for window in windows:
        if len(data) >= window:
            window_data = data.iloc[-window:]
            
            # Calculate scaling behavior
            scaling_metric = analyze_scaling_behavior(window_data)
            regime_scores.append(scaling_metric)
    
    if regime_scores:
        # Average regime score across scales
        regime_adjustment = np.mean(regime_scores)
        return np.clip(regime_adjustment, 0.5, 2.0)
    else:
        return 1.0

def analyze_scaling_behavior(data):
    """Analyze scaling behavior for regime classification"""
    
    prices = data['close'].values
    
    if len(prices) < 10:
        return 1.0
    
    # Calculate variance ratio at different lags
    returns = np.diff(np.log(prices))
    
    if len(returns) < 5:
        return 1.0
    
    # Simple scaling analysis using variance ratios
    var_1 = np.var(returns)
    var_2 = np.var(returns[::2]) if len(returns) >= 2 else var_1
    var_4 = np.var(returns[::4]) if len(returns) >= 4 else var_1
    
    if var_1 > 0:
        scaling_ratio_2 = var_2 / var_1 if var_2 > 0 else 1.0
        scaling_ratio_4 = var_4 / var_1 if var_4 > 0 else 1.0
        
        # Regime classification based on scaling behavior
        avg_scaling = (scaling_ratio_2 + scaling_ratio_4) / 2.0
        
        if avg_scaling > 1.2:
            return 1.5  # Trending regime
        elif avg_scaling < 0.8:
            return 0.7  # Mean-reverting regime
        else:
            return 1.0  # Random walk regime
    else:
        return 1.0

def estimate_hurst_simple(series):
    """Simple Hurst exponent estimation using variance scaling"""
    
    n = len(series)
    if n < 10:
        return 0.5
    
    # Calculate variances at different aggregation levels
    variances = []
    lags = [1, 2, 4, 8]
    
    for lag in lags:
        if n >= lag * 2:
            aggregated = series[::lag]
            if len(aggregated) > 1:
                variances.append(np.var(aggregated))
    
    if len(variances) < 2:
        return 0.5
    
    # Linear regression on log variances vs log lags
    lags_used = lags[:len(variances)]
    log_lags = np.log(lags_used)
    log_vars = np.log(variances)
    
    if len(log_lags) > 1 and np.std(log_lags) > 0:
        slope, _, _, _, _ = linregress(log_lags, log_vars)
        hurst = 1.0 + slope / 2.0
        return np.clip(hurst, 0.0, 1.0)
    else:
        return 0.5

def count_clusters(series):
    """Count the number of clusters in a boolean series"""
    
    if len(series) == 0:
        return 0
    
    clusters = 1 if series[0] else 0
    for i in range(1, len(series)):
        if series[i] and not series[i-1]:
            clusters += 1
    
    return clusters
