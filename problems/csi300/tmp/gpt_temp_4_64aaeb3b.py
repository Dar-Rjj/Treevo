import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import entropy

def heuristics_v2(df):
    """
    Nonlinear Price-Volume Entropy Factor combining price path complexity 
    with volume information content for return prediction.
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Measure Price Path Complexity
    # Calculate fractal dimension using Hurst exponent approximation
    def hurst_exponent(series, max_lag=20):
        """Approximate Hurst exponent using rescaled range analysis"""
        lags = range(2, min(max_lag + 1, len(series)))
        tau = []
        for lag in lags:
            # Create non-overlapping subseries
            subseries = [series.values[i:i+lag] for i in range(0, len(series)-lag, lag)]
            if len(subseries) < 2:
                continue
                
            # Calculate R/S for each subseries
            rs_values = []
            for sub in subseries:
                if len(sub) < 2:
                    continue
                # Detrend
                y = sub - np.mean(sub)
                # Cumulative deviate series
                z = np.cumsum(y)
                # Rescaled range
                r = np.max(z) - np.min(z)
                s = np.std(sub)
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            lags_log = [np.log(lag) for lag in lags[:len(tau)]]
            hurst = np.polyfit(lags_log, tau, 1)[0]
            return hurst
        return 1.0
    
    # Rolling fractal dimension calculation
    fractal_dim = pd.Series(index=data.index, dtype=float)
    window_size = 20
    
    for i in range(window_size, len(data)):
        window_data = data['close'].iloc[i-window_size:i]
        hurst = hurst_exponent(window_data)
        fractal_dim.iloc[i] = 2 - hurst  # Fractal dimension approximation
    
    # Count sign changes in price differences
    price_diff = data['close'].diff()
    sign_changes = (price_diff * price_diff.shift(1) < 0).rolling(window=10).sum()
    
    # Normalize and combine price complexity measures
    price_complexity = (fractal_dim.rolling(window=10).mean() * 
                       (1 + sign_changes.rolling(window=5).mean()))
    
    # 2. Assess Volume Information Content
    # Compute Shannon entropy of volume distribution
    def volume_entropy(volume_series, bins=10):
        """Calculate Shannon entropy of volume distribution"""
        if len(volume_series) < bins:
            return 0.0
        
        # Use percentiles for bin edges to handle non-stationarity
        percentiles = np.linspace(0, 100, bins + 1)
        bin_edges = np.percentile(volume_series, percentiles)
        
        # Handle duplicate edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
            
        hist, _ = np.histogram(volume_series, bins=bin_edges)
        hist = hist[hist > 0]  # Remove zero counts
        if len(hist) < 2:
            return 0.0
            
        prob = hist / np.sum(hist)
        return entropy(prob)
    
    # Rolling volume entropy
    volume_entropy_vals = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        volume_window = data['volume'].iloc[i-20:i]
        volume_entropy_vals.iloc[i] = volume_entropy(volume_window)
    
    # Identify volume regime shifts using CUSUM
    volume_mean = data['volume'].rolling(window=20).mean()
    volume_std = data['volume'].rolling(window=20).std()
    volume_zscore = (data['volume'] - volume_mean) / volume_std
    regime_shifts = (volume_zscore.abs() > 2).rolling(window=5).sum()
    
    # Combine volume information measures
    volume_info = (volume_entropy_vals * 
                  (1 + regime_shifts.rolling(window=5).mean()))
    
    # 3. Generate Entropy Convergence Signal
    # Multiply price complexity by volume information
    entropy_signal = price_complexity * volume_info
    
    # Invert relationship for predictive power (higher entropy -> lower expected returns)
    factor = -entropy_signal.rolling(window=5).mean()
    
    # Handle NaN values
    factor = factor.fillna(0)
    
    return factor
