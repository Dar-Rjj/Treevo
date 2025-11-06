import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Price-Volume Divergence with Fractal Market Structure
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price Fractals
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using R/S analysis"""
        lags = range(2, min(window, len(series))//2)
        tau = []
        for lag in lags:
            if len(series) >= lag*2:
                # Create non-overlapping windows
                rs_values = []
                for i in range(0, len(series)-lag, lag):
                    segment = series.iloc[i:i+lag]
                    if len(segment) == lag:
                        mean_segment = segment.mean()
                        cum_dev = (segment - mean_segment).cumsum()
                        r = cum_dev.max() - cum_dev.min()
                        s = segment.std()
                        if s > 0:
                            rs_values.append(r/s)
                if rs_values:
                    tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            lags_arr = np.log(lags[:len(tau)])
            hurst = np.polyfit(lags_arr, tau, 1)[0]
            return hurst
        return 0.5
    
    # Price fractal calculations
    def calculate_price_fractals(data, window):
        """Calculate price-based fractal metrics"""
        results = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            if i >= window:
                close_data = data['close'].iloc[i-window:i]
                high_data = data['high'].iloc[i-window:i]
                low_data = data['low'].iloc[i-window:i]
                
                # Hurst exponent for price persistence
                hurst_price = hurst_exponent(close_data, window//2)
                
                # Fractal scaling ratio (high-low range complexity)
                price_ranges = (high_data - low_data) / data['close'].iloc[i-window]
                range_complexity = price_ranges.std() / price_ranges.mean() if price_ranges.mean() > 0 else 0
                
                # Price path complexity (sum of absolute changes)
                price_changes = close_data.diff().abs().sum()
                normalized_complexity = price_changes / (close_data.max() - close_data.min()) if (close_data.max() - close_data.min()) > 0 else 0
                
                # Combined fractal score
                fractal_score = (hurst_price + range_complexity + normalized_complexity) / 3
                results.iloc[i] = fractal_score
        
        return results.ffill().fillna(0.5)
    
    # Volume fractal calculations
    def calculate_volume_fractals(data, window):
        """Calculate volume-based fractal metrics"""
        results = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            if i >= window:
                volume_data = data['volume'].iloc[i-window:i]
                amount_data = data['amount'].iloc[i-window:i]
                
                # Volume Hurst exponent
                hurst_volume = hurst_exponent(volume_data, window//2)
                
                # Volume correlation decay (autocorrelation at lag 1)
                if len(volume_data) > 1:
                    volume_autocorr = volume_data.autocorr(lag=1)
                    if pd.isna(volume_autocorr):
                        volume_autocorr = 0
                else:
                    volume_autocorr = 0
                
                # Volume clustering intensity (variance-to-mean ratio)
                volume_clustering = volume_data.var() / volume_data.mean() if volume_data.mean() > 0 else 0
                
                # Combined volume fractal score
                volume_fractal = (hurst_volume + abs(volume_autocorr) + min(volume_clustering, 5)/5) / 3
                results.iloc[i] = volume_fractal
        
        return results.ffill().fillna(0.5)
    
    # Market microstructure patterns
    def calculate_microstructure(data):
        """Calculate intraday market microstructure metrics"""
        # Price efficiency: High-Low range relative to Close-Open
        price_efficiency = ((data['high'] - data['low']) / (abs(data['close'] - data['open']) + 1e-8)).replace([np.inf, -np.inf], 1)
        
        # Price rejection patterns (shadows relative to body)
        upper_shadow = (data['high'] - np.maximum(data['open'], data['close'])) / (data['high'] - data['low'] + 1e-8)
        lower_shadow = (np.minimum(data['open'], data['close']) - data['low']) / (data['high'] - data['low'] + 1e-8)
        rejection_strength = abs(upper_shadow - lower_shadow)
        
        # Volume-price correlation at different scales
        volume_price_corr = data['volume'].rolling(window=5).corr(data['close'].pct_change().abs())
        
        # Combined microstructure score
        microstructure = (price_efficiency.rolling(5).mean() + 
                         rejection_strength.rolling(5).mean() + 
                         volume_price_corr.fillna(0)) / 3
        
        return microstructure.fillna(0)
    
    # Calculate fractal components
    price_fractal_short = calculate_price_fractals(data, 5)
    price_fractal_medium = calculate_price_fractals(data, 10)
    price_fractal_long = calculate_price_fractals(data, 20)
    
    volume_fractal_short = calculate_volume_fractals(data, 5)
    volume_fractal_medium = calculate_volume_fractals(data, 10)
    volume_fractal_long = calculate_volume_fractals(data, 20)
    
    # Price-Volume Fractal Divergence
    pv_divergence_short = price_fractal_short - volume_fractal_short
    pv_divergence_medium = price_fractal_medium - volume_fractal_medium
    pv_divergence_long = price_fractal_long - volume_fractal_long
    
    # Multi-timeframe divergence consistency
    divergence_consistency = (pv_divergence_short.rolling(3).std().fillna(0) + 
                            pv_divergence_medium.rolling(5).std().fillna(0) + 
                            pv_divergence_long.rolling(8).std().fillna(0)) / 3
    
    # Market microstructure
    microstructure = calculate_microstructure(data)
    
    # Generate final alpha signal with regime-dependent weighting
    # Regime detection based on fractal dimensions
    price_regime = price_fractal_long.rolling(10).mean()
    volume_regime = volume_fractal_long.rolling(10).mean()
    
    # High fractal dimension = trending, low = mean-reverting
    regime_adjustment = np.where(price_regime > 0.6, 1.0, 
                                np.where(price_regime < 0.4, -1.0, 0.0))
    
    # Combine signals with regime adjustment
    short_term_signal = pv_divergence_short * microstructure
    medium_term_signal = pv_divergence_medium * (1 - divergence_consistency)
    long_term_signal = pv_divergence_long * regime_adjustment
    
    # Final alpha factor with multi-scale integration
    alpha_factor = (short_term_signal.rolling(3).mean().fillna(0) * 0.4 +
                   medium_term_signal.rolling(5).mean().fillna(0) * 0.4 +
                   long_term_signal.rolling(8).mean().fillna(0) * 0.2)
    
    return alpha_factor.fillna(0)
