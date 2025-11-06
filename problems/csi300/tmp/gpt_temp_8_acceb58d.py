import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Fractal Market Microstructure Factor combining price and volume fractal dynamics
    with geometric market flow analysis for alpha generation.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Fractal Geometry Analysis
    # Calculate fractal dimension using High-Low-Close price ranges
    data['price_range'] = data['high'] - data['low']
    data['price_mid'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Compute Hurst-like exponent using rescaled range analysis
    def hurst_exponent(series, window=20):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
            window_data = series.iloc[i-window:i]
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative = deviations.cumsum()
            r = cumulative.max() - cumulative.min()
            s = window_data.std()
            if s > 0:
                hurst_values.append(np.log(r/s) / np.log(window))
            else:
                hurst_values.append(np.nan)
        return pd.Series(hurst_values, index=series.index)
    
    data['hurst_price'] = hurst_exponent(data['close'], window=20)
    
    # Volume Fractal Dynamics
    # Calculate volume fractal dimension using volume clustering
    def volume_fractal_dimension(volume_series, window=15):
        fractal_dims = []
        for i in range(len(volume_series)):
            if i < window:
                fractal_dims.append(np.nan)
                continue
            window_vol = volume_series.iloc[i-window:i]
            # Simple box-counting inspired approach
            vol_range = window_vol.max() - window_vol.min()
            if vol_range > 0:
                # Count significant volume clusters
                threshold = window_vol.median()
                clusters = (window_vol > threshold).astype(int)
                cluster_changes = clusters.diff().fillna(0).abs().sum()
                if cluster_changes > 0:
                    fractal_dim = np.log(cluster_changes) / np.log(window)
                    fractal_dims.append(fractal_dim)
                else:
                    fractal_dims.append(np.nan)
            else:
                fractal_dims.append(np.nan)
        return pd.Series(fractal_dims, index=volume_series.index)
    
    data['volume_fractal_dim'] = volume_fractal_dimension(data['volume'], window=15)
    
    # Microstructure Fractal Alignment
    # Compute correlation between price and volume fractal patterns
    def fractal_alignment(price_series, volume_series, window=10):
        alignment = []
        for i in range(len(price_series)):
            if i < window:
                alignment.append(np.nan)
                continue
            price_window = price_series.iloc[i-window:i]
            volume_window = volume_series.iloc[i-window:i]
            
            # Normalize both series
            price_norm = (price_window - price_window.mean()) / (price_window.std() + 1e-8)
            volume_norm = (volume_window - volume_window.mean()) / (volume_window.std() + 1e-8)
            
            # Calculate phase synchronization using cross-correlation
            correlation = price_norm.corr(volume_norm)
            if not np.isnan(correlation):
                alignment.append(correlation)
            else:
                alignment.append(0)
        return pd.Series(alignment, index=price_series.index)
    
    data['fractal_alignment'] = fractal_alignment(data['close'], data['volume'], window=10)
    
    # Geometric Market Flow Analysis
    # Calculate price trajectory curvature
    def price_curvature(close_series, window=5):
        curvature = []
        for i in range(len(close_series)):
            if i < window:
                curvature.append(np.nan)
                continue
            window_prices = close_series.iloc[i-window:i]
            if len(window_prices) < 3:
                curvature.append(np.nan)
                continue
            
            # Simple curvature approximation using second derivative
            x = np.arange(len(window_prices))
            y = window_prices.values
            try:
                coeffs = np.polyfit(x, y, 2)
                # Curvature is related to second derivative (2*a)
                curvature.append(abs(2 * coeffs[0]))
            except:
                curvature.append(np.nan)
        return pd.Series(curvature, index=close_series.index)
    
    data['price_curvature'] = price_curvature(data['close'], window=5)
    
    # Volume geometric flow - divergence calculation
    def volume_divergence(volume_series, price_series, window=8):
        divergence = []
        for i in range(len(volume_series)):
            if i < window:
                divergence.append(np.nan)
                continue
            vol_window = volume_series.iloc[i-window:i]
            price_window = price_series.iloc[i-window:i]
            
            # Calculate volume gradient and price gradient
            vol_gradient = np.gradient(vol_window.values)
            price_gradient = np.gradient(price_window.values)
            
            # Simple divergence measure
            if len(vol_gradient) > 1 and len(price_gradient) > 1:
                # Correlation between volume change and price change directions
                direction_corr = np.corrcoef(vol_gradient, price_gradient)[0,1]
                if not np.isnan(direction_corr):
                    divergence.append(direction_corr)
                else:
                    divergence.append(0)
            else:
                divergence.append(0)
        return pd.Series(divergence, index=volume_series.index)
    
    data['volume_divergence'] = volume_divergence(data['volume'], data['close'], window=8)
    
    # Fractal Microstructure Signal Generation
    # Combine all components with appropriate weights
    def generate_fractal_signal(data):
        signals = []
        
        # Normalize components
        hurst_norm = (data['hurst_price'] - data['hurst_price'].rolling(50).mean()) / (data['hurst_price'].rolling(50).std() + 1e-8)
        vol_fractal_norm = (data['volume_fractal_dim'] - data['volume_fractal_dim'].rolling(50).mean()) / (data['volume_fractal_dim'].rolling(50).std() + 1e-8)
        alignment_norm = data['fractal_alignment'].fillna(0)
        curvature_norm = (data['price_curvature'] - data['price_curvature'].rolling(50).mean()) / (data['price_curvature'].rolling(50).std() + 1e-8)
        divergence_norm = data['volume_divergence'].fillna(0)
        
        for i in range(len(data)):
            if i < 50:
                signals.append(np.nan)
                continue
            
            # Composite signal with momentum from changes
            hurst_val = hurst_norm.iloc[i] if not np.isnan(hurst_norm.iloc[i]) else 0
            vol_fractal_val = vol_fractal_norm.iloc[i] if not np.isnan(vol_fractal_norm.iloc[i]) else 0
            alignment_val = alignment_norm.iloc[i]
            curvature_val = curvature_norm.iloc[i] if not np.isnan(curvature_norm.iloc[i]) else 0
            divergence_val = divergence_norm.iloc[i]
            
            # Fractal momentum - changes in fractal dimensions
            if i > 1:
                hurst_momentum = hurst_norm.iloc[i] - hurst_norm.iloc[i-1] if not np.isnan(hurst_norm.iloc[i]) and not np.isnan(hurst_norm.iloc[i-1]) else 0
                vol_fractal_momentum = vol_fractal_norm.iloc[i] - vol_fractal_norm.iloc[i-1] if not np.isnan(vol_fractal_norm.iloc[i]) and not np.isnan(vol_fractal_norm.iloc[i-1]) else 0
            else:
                hurst_momentum = 0
                vol_fractal_momentum = 0
            
            # Final composite signal
            signal = (
                0.3 * hurst_val + 
                0.25 * vol_fractal_val + 
                0.2 * alignment_val + 
                0.15 * curvature_val + 
                0.1 * divergence_val +
                0.05 * hurst_momentum +
                0.05 * vol_fractal_momentum
            )
            
            signals.append(signal)
        
        return pd.Series(signals, index=data.index)
    
    # Generate the final alpha factor
    alpha_factor = generate_fractal_signal(data)
    
    return alpha_factor
