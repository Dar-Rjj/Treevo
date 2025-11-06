import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Scale Fractal Market Microstructure Alpha Factor
    Combines Hurst exponent analysis, volume-time fractality, and multi-timeframe
    fractal regime detection to generate adaptive trading signals.
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate typical price for fractal analysis
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Multi-timeframe window sizes (days)
    windows = [5, 20, 60]
    
    for i in range(max(windows), len(df)):
        current_data = df.iloc[:i+1]
        current_date = df.index[i]
        
        # 1. Multi-timeframe Hurst Exponent Calculation
        hurst_values = []
        for window in windows:
            if i >= window:
                window_data = typical_price.iloc[i-window+1:i+1]
                
                # Calculate rescaled range (R/S) for Hurst estimation
                if len(window_data) > 1:
                    # Calculate mean and deviations
                    mean_val = window_data.mean()
                    deviations = window_data - mean_val
                    
                    # Calculate cumulative deviations
                    Z = deviations.cumsum()
                    
                    # Calculate range
                    R = Z.max() - Z.min()
                    
                    # Calculate standard deviation
                    S = window_data.std()
                    
                    # Avoid division by zero
                    if S > 0:
                        RS = R / S
                        # Simple Hurst estimation (log(R/S) ~ H * log(n))
                        if RS > 0:
                            hurst = np.log(RS) / np.log(len(window_data))
                            hurst_values.append(hurst)
        
        # 2. Volume-Time Fractality Analysis
        volume_fractality = 0
        if i >= 20:
            volume_data = df['volume'].iloc[i-19:i+1]
            price_data = typical_price.iloc[i-19:i+1]
            
            # Calculate volume clustering fractality
            volume_changes = volume_data.pct_change().dropna()
            if len(volume_changes) > 5:
                # Simple fractal dimension estimation via variance scaling
                variances = []
                scales = [1, 2, 4]
                
                for scale in scales:
                    if len(volume_changes) >= scale:
                        scaled_data = volume_changes.rolling(scale).mean().dropna()
                        if len(scaled_data) > 0:
                            variances.append(scaled_data.var())
                
                if len(variances) >= 2 and variances[0] > 0:
                    # Estimate fractal dimension from variance scaling
                    volume_fractality = np.log(variances[-1] / variances[0]) / np.log(scales[-1] / scales[0])
        
        # 3. Price Path Roughness Measurement
        path_roughness = 0
        if i >= 10:
            recent_prices = typical_price.iloc[i-9:i+1]
            price_changes = recent_prices.pct_change().dropna()
            
            if len(price_changes) > 0:
                # Calculate roughness as normalized absolute changes
                abs_changes = np.abs(price_changes)
                path_roughness = abs_changes.std() / abs_changes.mean() if abs_changes.mean() > 0 else 0
        
        # 4. Fractal Regime Classification
        fractal_regime = 0
        if hurst_values:
            avg_hurst = np.mean(hurst_values)
            
            # Classify regimes based on Hurst exponent
            if avg_hurst > 0.6:
                fractal_regime = 1  # Trending regime
            elif avg_hurst < 0.4:
                fractal_regime = -1  # Mean-reverting regime
            else:
                fractal_regime = 0  # Random walk regime
        
        # 5. Multi-Scale Fractal Signal Combination
        alpha_signal = 0
        
        if hurst_values and len(hurst_values) >= 2:
            # Base signal from Hurst exponent
            base_signal = np.mean(hurst_values) - 0.5  # Deviation from random walk
            
            # Volume fractality adjustment
            volume_weight = 0.3 * volume_fractality
            
            # Path roughness adjustment (higher roughness suggests mean reversion)
            roughness_weight = -0.2 * path_roughness
            
            # Combine signals with regime-based weighting
            if fractal_regime == 1:  # Trending - emphasize momentum
                alpha_signal = base_signal * 1.5 + volume_weight * 0.5 + roughness_weight * 0.2
            elif fractal_regime == -1:  # Mean-reverting - emphasize contrarian
                alpha_signal = base_signal * 0.7 + volume_weight * 0.3 + roughness_weight * 0.8
            else:  # Random walk - reduced signal strength
                alpha_signal = base_signal * 0.5 + volume_weight * 0.2 + roughness_weight * 0.3
        
        result.iloc[i] = alpha_signal
    
    # Forward fill any NaN values at the beginning
    result = result.fillna(method='ffill').fillna(0)
    
    return result
