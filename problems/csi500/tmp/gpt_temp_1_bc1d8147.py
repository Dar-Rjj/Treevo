import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics with Regime Transition Detection
    Multi-scale fractal analysis combining price and volume dynamics
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using R/S analysis"""
        if len(series) < window:
            return np.nan
        
        # Calculate log returns
        returns = np.log(series / series.shift(1)).dropna()
        if len(returns) < window:
            return np.nan
            
        # R/S analysis
        max_lag = min(window // 2, 50)
        lags = range(2, max_lag + 1)
        rs_values = []
        
        for lag in lags:
            # Calculate rescaled range for each lag
            n_splits = len(returns) // lag
            if n_splits < 2:
                continue
                
            rs_lag = []
            for i in range(n_splits):
                segment = returns[i*lag:(i+1)*lag]
                if len(segment) < 2:
                    continue
                    
                # Calculate mean and cumulative deviations
                mean_segment = segment.mean()
                deviations = segment - mean_segment
                cumulative = deviations.cumsum()
                
                # Range and standard deviation
                R = cumulative.max() - cumulative.min()
                S = segment.std()
                
                if S > 0:
                    rs_lag.append(R / S)
            
            if rs_lag:
                rs_values.append(np.log(np.mean(rs_lag)))
            else:
                rs_values.append(np.nan)
        
        # Fit linear regression to log(R/S) vs log(lag)
        valid_lags = [l for l, rs in zip(lags, rs_values) if not np.isnan(rs)]
        valid_rs = [rs for rs in rs_values if not np.isnan(rs)]
        
        if len(valid_lags) >= 3:
            slope, _, _, _, _ = linregress(np.log(valid_lags), valid_rs)
            return slope
        else:
            return np.nan
    
    def volume_fractal_dimension(volume_series, window):
        """Calculate fractal dimension from volume series using box counting method"""
        if len(volume_series) < window:
            return np.nan
            
        segment = volume_series[-window:]
        if segment.std() == 0:
            return 1.0
            
        # Simplified box counting approach
        max_val = segment.max()
        min_val = segment.min()
        range_val = max_val - min_val
        
        if range_val == 0:
            return 1.0
            
        # Calculate roughness measure
        normalized_vol = (segment - min_val) / range_val
        roughness = normalized_vol.diff().abs().mean()
        
        # Convert to fractal dimension estimate (1.0 = smooth, 2.0 = very rough)
        fractal_dim = 1.0 + roughness
        return min(fractal_dim, 2.0)
    
    def multi_fractal_spectrum(high_series, low_series, window):
        """Calculate multi-fractal spectrum width from high-low range"""
        if len(high_series) < window:
            return np.nan
            
        segment_high = high_series[-window:]
        segment_low = low_series[-window:]
        
        # Calculate daily ranges
        daily_ranges = segment_high - segment_low
        if daily_ranges.std() == 0:
            return 0.0
            
        # Normalize ranges
        norm_ranges = (daily_ranges - daily_ranges.mean()) / daily_ranges.std()
        
        # Calculate moments for multi-fractal spectrum estimation
        q_values = [-2, -1, 1, 2]
        moments = []
        
        for q in q_values:
            if q != 0:
                moment = (np.abs(norm_ranges) ** q).mean()
                moments.append(np.log(moment) if moment > 0 else 0)
            else:
                moments.append(0)
        
        # Estimate spectrum width from moments
        if len(moments) >= 2:
            spectrum_width = max(moments) - min(moments)
            return spectrum_width
        else:
            return 0.0
    
    def information_entropy(price_series, window):
        """Calculate information entropy from price movements"""
        if len(price_series) < window:
            return np.nan
            
        returns = np.log(price_series / price_series.shift(1)).dropna()
        if len(returns) < window:
            return np.nan
            
        segment = returns[-window:]
        
        # Symbolic dynamics: categorize returns into 3 states
        threshold = segment.std()
        states = pd.cut(segment, bins=[-np.inf, -threshold, threshold, np.inf], labels=[-1, 0, 1])
        
        # Calculate probability distribution
        state_counts = states.value_counts(normalize=True)
        
        # Calculate entropy
        entropy = -sum(p * np.log(p) for p in state_counts if p > 0)
        return entropy
    
    # Initialize result series
    alpha_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate fractal features for each day
    for i in range(20, len(data)):
        current_data = data.iloc[:i+1]
        
        # Multi-scale price fractal dimensions
        hurst_5 = hurst_exponent(current_data['close'], 5)
        hurst_10 = hurst_exponent(current_data['close'], 10)
        hurst_20 = hurst_exponent(current_data['close'], 20)
        
        # Volume fractal dimensions
        vol_fractal_5 = volume_fractal_dimension(current_data['volume'], 5)
        vol_fractal_10 = volume_fractal_dimension(current_data['volume'], 10)
        vol_fractal_20 = volume_fractal_dimension(current_data['volume'], 20)
        
        # Multi-fractal spectrum
        multi_fractal = multi_fractal_spectrum(current_data['high'], current_data['low'], 10)
        
        # Information entropy
        price_entropy = information_entropy(current_data['close'], 10)
        volume_entropy = information_entropy(current_data['volume'], 10)
        
        # Price-volume fractal correlation
        if not np.isnan(hurst_10) and not np.isnan(vol_fractal_10):
            fractal_corr = 1.0 - abs(hurst_10 - vol_fractal_10)
        else:
            fractal_corr = 0.5
        
        # Multi-timeframe fractal alignment
        hurst_values = [hurst_5, hurst_10, hurst_20]
        valid_hurst = [h for h in hurst_values if not np.isnan(h)]
        if len(valid_hurst) >= 2:
            hurst_alignment = 1.0 - np.std(valid_hurst)
        else:
            hurst_alignment = 0.0
        
        # Information flow asymmetry
        if not np.isnan(price_entropy) and not np.isnan(volume_entropy):
            info_asymmetry = abs(price_entropy - volume_entropy)
        else:
            info_asymmetry = 0.0
        
        # Combined alpha factor
        alpha = 0.0
        
        # Fractal regime timing component
        if not np.isnan(hurst_10):
            # Mean-reverting markets (H < 0.5) vs trending markets (H > 0.5)
            regime_signal = (hurst_10 - 0.5) * 2  # Normalize to [-1, 1]
            alpha += regime_signal * 0.3
        
        # Fractal correlation component
        alpha += (fractal_corr - 0.5) * 0.4
        
        # Multi-fractal spectrum component
        alpha += multi_fractal * 0.2
        
        # Information asymmetry component
        alpha -= info_asymmetry * 0.1
        
        # Store alpha value
        alpha_values.iloc[i] = alpha
    
    # Fill initial NaN values with 0
    alpha_values = alpha_values.fillna(0)
    
    return alpha_values
