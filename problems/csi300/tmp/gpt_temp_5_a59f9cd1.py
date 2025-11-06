import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Pressure Convergence Divergence factor
    Combines price fractality, volume fractality, and bidirectional pressure analysis
    to identify trend strength and reversal potential.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Price Fractal Dimension (Hurst exponent approximation)
    def hurst_exponent(series, window=10):
        """Calculate Hurst exponent approximation using rescaled range analysis"""
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                hurst_values.append(np.nan)
                continue
                
            # Calculate mean
            mean_val = window_data.mean()
            
            # Calculate cumulative deviations
            deviations = window_data - mean_val
            cumulative_deviations = deviations.cumsum()
            
            # Calculate range
            R = cumulative_deviations.max() - cumulative_deviations.min()
            
            # Calculate standard deviation
            S = window_data.std()
            
            # Avoid division by zero
            if S == 0:
                hurst_values.append(np.nan)
                continue
                
            # Calculate Hurst
            hurst = np.log(R / S) / np.log(window)
            hurst_values.append(hurst)
            
        return pd.Series(hurst_values, index=series.index)
    
    # Calculate price Hurst using close prices
    price_hurst = hurst_exponent(data['close'], window=10)
    
    # 2. Analyze Volume Fractality
    def volume_fractality(volume_series, window=5):
        """Calculate volume clustering patterns"""
        fractality_values = []
        for i in range(len(volume_series)):
            if i < window:
                fractality_values.append(np.nan)
                continue
                
            window_data = volume_series.iloc[i-window+1:i+1]
            
            # Calculate volume persistence (autocorrelation at lag 1)
            if len(window_data) > 1:
                autocorr = window_data.autocorr(lag=1)
                if pd.isna(autocorr):
                    fractality_values.append(0)
                else:
                    fractality_values.append(abs(autocorr))
            else:
                fractality_values.append(0)
                
        return pd.Series(fractality_values, index=volume_series.index)
    
    volume_fractal = volume_fractality(data['volume'], window=5)
    
    # 3. Calculate Bidirectional Pressure
    def calculate_pressure(data, window=3):
        """Calculate upward and downward pressure"""
        upward_pressure = []
        downward_pressure = []
        pressure_ratio = []
        
        for i in range(len(data)):
            if i < window:
                upward_pressure.append(np.nan)
                downward_pressure.append(np.nan)
                pressure_ratio.append(np.nan)
                continue
                
            # Upward pressure: High-Close momentum
            high_close_momentum = []
            for j in range(i-window+1, i+1):
                if j >= 0:
                    momentum = (data['high'].iloc[j] - data['close'].iloc[j]) / data['close'].iloc[j]
                    high_close_momentum.append(momentum)
            
            upward = np.mean(high_close_momentum) if high_close_momentum else 0
            
            # Downward pressure: Close-Low momentum
            close_low_momentum = []
            for j in range(i-window+1, i+1):
                if j >= 0:
                    momentum = (data['close'].iloc[j] - data['low'].iloc[j]) / data['close'].iloc[j]
                    close_low_momentum.append(momentum)
            
            downward = np.mean(close_low_momentum) if close_low_momentum else 0
            
            upward_pressure.append(upward)
            downward_pressure.append(downward)
            
            # Pressure imbalance ratio
            if upward + downward == 0:
                pressure_ratio.append(0)
            else:
                pressure_ratio.append((upward - downward) / (upward + downward))
                
        return (pd.Series(upward_pressure, index=data.index),
                pd.Series(downward_pressure, index=data.index),
                pd.Series(pressure_ratio, index=data.index))
    
    upward_pressure, downward_pressure, pressure_ratio = calculate_pressure(data, window=3)
    
    # 4. Detect Convergence/Divergence Patterns
    def fractal_pressure_signal(price_hurst, volume_fractal, pressure_ratio, volume_series):
        """Generate adaptive signal based on fractal pressure convergence/divergence"""
        signals = []
        
        for i in range(len(price_hurst)):
            if pd.isna(price_hurst.iloc[i]) or pd.isna(volume_fractal.iloc[i]) or pd.isna(pressure_ratio.iloc[i]):
                signals.append(np.nan)
                continue
            
            # Normalize inputs
            hurst_norm = (price_hurst.iloc[i] - 0.5) * 2  # Center around 0, scale to [-1,1]
            vol_fractal_norm = volume_fractal.iloc[i] * 2 - 1  # Scale to [-1,1]
            pressure_norm = pressure_ratio.iloc[i]
            
            # High fractality + pressure divergence: weak trends (reversal potential)
            if hurst_norm > 0.3 and abs(pressure_norm) > 0.2:
                # High randomness with strong pressure imbalance suggests reversal
                signal_strength = -pressure_norm * hurst_norm
                
            # Low fractality + pressure convergence: strong trends (continuation)
            elif hurst_norm < 0.1 and abs(pressure_norm) < 0.1:
                # Low randomness with balanced pressure suggests trend continuation
                signal_strength = hurst_norm * -10  # Negative because low Hurst = trending
                
            else:
                # Mixed regime - use volume fractality for confirmation
                if vol_fractal_norm > 0:
                    # High volume persistence confirms signals
                    signal_strength = pressure_norm * (1 + vol_fractal_norm)
                else:
                    # Low volume persistence reduces signal strength
                    signal_strength = pressure_norm * (1 + vol_fractal_norm) * 0.5
            
            # Apply volume-based weighting
            if i > 0:
                vol_weight = volume_series.iloc[i] / volume_series.iloc[max(0, i-5):i+1].mean()
                if not pd.isna(vol_weight):
                    signal_strength *= min(vol_weight, 2)  # Cap volume weight at 2x
                    
            signals.append(signal_strength)
            
        return pd.Series(signals, index=price_hurst.index)
    
    # Generate final factor
    factor = fractal_pressure_signal(price_hurst, volume_fractal, pressure_ratio, data['volume'])
    
    # Normalize the factor
    if factor.notna().sum() > 0:
        factor = (factor - factor.mean()) / factor.std()
    
    return factor
