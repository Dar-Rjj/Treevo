import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Nonlinear Price-Volume Entropy Factor
    Combines price path complexity with volume information content
    High entropy suggests mean reversion, low entropy suggests momentum
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price Path Complexity
    # Calculate Hurst exponent approximation using 10-day window
    def hurst_approximation(series, window=10):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i]
            # Rescaled range method approximation
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            Z = deviations.cumsum()
            R = Z.max() - Z.min()
            S = window_data.std()
            
            if S > 0:
                hurst = np.log(R/S) / np.log(window)
            else:
                hurst = 0.5
            hurst_values.append(hurst)
        
        return pd.Series(hurst_values, index=series.index)
    
    # Use close prices for Hurst calculation
    hurst_close = hurst_approximation(data['close'], window=10)
    
    # Count sign changes in price differences (directional randomness)
    price_diff = data['close'].diff()
    sign_changes = (price_diff * price_diff.shift(1) < 0).rolling(window=10).sum()
    
    # Normalize and combine price complexity measures
    price_complexity = (hurst_close.fillna(0.5) + 
                       (sign_changes.fillna(0) / 10)) / 2
    
    # 2. Volume Information Content
    # Calculate Shannon entropy of volume distribution using 15-day rolling percentiles
    def volume_entropy(volume_series, window=15, bins=5):
        entropy_values = []
        for i in range(len(volume_series)):
            if i < window:
                entropy_values.append(np.nan)
                continue
                
            window_volumes = volume_series.iloc[i-window:i]
            # Calculate percentiles to create bins
            percentiles = np.percentile(window_volumes, [20, 40, 60, 80])
            
            # Count observations in each bin
            counts = np.zeros(bins)
            for vol in window_volumes:
                if vol <= percentiles[0]:
                    counts[0] += 1
                elif vol <= percentiles[1]:
                    counts[1] += 1
                elif vol <= percentiles[2]:
                    counts[2] += 1
                elif vol <= percentiles[3]:
                    counts[3] += 1
                else:
                    counts[4] += 1
            
            # Calculate Shannon entropy
            probabilities = counts / window
            probabilities = probabilities[probabilities > 0]  # Remove zeros for log
            entropy = -np.sum(probabilities * np.log(probabilities))
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=volume_series.index)
    
    volume_entropy_series = volume_entropy(data['volume'], window=15)
    
    # Detect volume clustering anomalies using Z-score of volume changes
    volume_change = data['volume'].pct_change()
    volume_zscore = volume_change.rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, 
        raw=False
    )
    volume_anomaly = volume_zscore.abs().fillna(0)
    
    # Normalize and combine volume information measures
    max_entropy = np.log(5)  # Maximum entropy for 5 bins
    volume_information = ((volume_entropy_series.fillna(0) / max_entropy) + 
                         volume_anomaly.fillna(0)) / 2
    
    # 3. Generate Entropy Convergence Signal
    # Multiply price complexity by volume information
    entropy_product = price_complexity * volume_information
    
    # Invert relationship: high entropy → mean reversion, low entropy → momentum
    # Use sigmoid function to create smooth transition
    factor = 1 / (1 + np.exp(3 * (entropy_product - 0.5)))
    
    # Ensure proper indexing and handle NaN values
    factor_series = pd.Series(factor, index=data.index).fillna(0.5)
    
    return factor_series
