import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Fractal Dimension Divergence combined with multiple alpha signals
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    # Price Fractal Dimension Calculation
    def calculate_hurst_exponent(series, max_lag=20):
        """Calculate Hurst exponent using R/S analysis"""
        if len(series) < max_lag * 2:
            return 0.5
        
        lags = range(2, min(max_lag, len(series)//2))
        tau = []
        
        for lag in lags:
            # Calculate rescaled range
            series_lag = []
            for i in range(0, len(series) - lag, lag):
                sub_series = series[i:i + lag]
                if len(sub_series) < lag:
                    continue
                mean_sub = np.mean(sub_series)
                cumulative_deviation = sub_series - mean_sub
                cumulative_sum = np.cumsum(cumulative_deviation)
                r = np.max(cumulative_sum) - np.min(cumulative_sum)
                s = np.std(sub_series)
                if s > 0:
                    series_lag.append(r / s)
            
            if series_lag:
                tau.append(np.log(np.mean(series_lag)))
        
        if len(tau) > 1:
            lags_log = [np.log(lag) for lag in lags[:len(tau)]]
            hurst = np.polyfit(lags_log, tau, 1)[0]
            return hurst
        return 0.5
    
    # Multi-scale fractal measure using high-low ranges
    def calculate_price_fractal(data, window=50):
        """Calculate price fractal dimension using box-counting method"""
        if len(data) < window:
            return pd.Series([1.5] * len(data), index=data.index)
        
        fractal_dim = []
        for i in range(len(data)):
            if i < window:
                fractal_dim.append(1.5)
                continue
            
            window_data = data.iloc[i-window:i]
            high_low_range = (window_data['high'] - window_data['low']).abs()
            
            if high_low_range.std() == 0:
                fractal_dim.append(1.5)
                continue
            
            # Simplified box-counting approach
            price_range = window_data['high'].max() - window_data['low'].min()
            if price_range == 0:
                fractal_dim.append(1.5)
                continue
            
            # Calculate fractal dimension approximation
            avg_range = high_low_range.mean()
            fractal = 2 - np.log(avg_range / price_range) / np.log(window)
            fractal_dim.append(max(1.0, min(2.0, fractal)))
        
        return pd.Series(fractal_dim, index=data.index)
    
    # Volume fractal dimension using entropy measures
    def calculate_volume_fractal(data, window=30):
        """Calculate volume fractal dimension using entropy-based approach"""
        if len(data) < window:
            return pd.Series([1.0] * len(data), index=data.index)
        
        volume_fractal = []
        for i in range(len(data)):
            if i < window:
                volume_fractal.append(1.0)
                continue
            
            window_volume = data['volume'].iloc[i-window:i]
            if window_volume.std() == 0:
                volume_fractal.append(1.0)
                continue
            
            # Calculate sample entropy as fractal dimension proxy
            volume_diff = window_volume.diff().dropna()
            if len(volume_diff) < 2:
                volume_fractal.append(1.0)
                continue
            
            # Simplified entropy calculation
            volume_entropy = stats.entropy(np.histogram(volume_diff, bins=5)[0] + 1e-10)
            fractal_dim = 1.0 + volume_entropy / np.log(10)  # Normalize
            volume_fractal.append(max(0.5, min(2.0, fractal_dim)))
        
        return pd.Series(volume_fractal, index=data.index)
    
    # Calculate fractal dimensions
    price_fractal = calculate_price_fractal(data, window=50)
    volume_fractal = calculate_volume_fractal(data, window=30)
    
    # Fractal Dimension Divergence
    fractal_ratio = price_fractal / (volume_fractal + 1e-10)
    fractal_divergence = np.log(fractal_ratio + 1e-10)
    
    # Rate of change for regime detection
    fractal_roc = fractal_divergence.diff(5).fillna(0)
    
    # Acceleration-Deceleration Asymmetry
    def calculate_acceleration_profile(data, window=20):
        """Calculate price acceleration asymmetry"""
        close_prices = data['close']
        
        # First derivative (velocity)
        velocity = close_prices.diff(3).rolling(window=5).mean()
        
        # Second derivative (acceleration)
        acceleration = velocity.diff(3).rolling(window=5).mean()
        
        # Third derivative (jerk)
        jerk = acceleration.diff(3).rolling(window=5).mean()
        
        # Calculate asymmetry
        pos_acceleration = acceleration.where(acceleration > 0, 0)
        neg_acceleration = acceleration.where(acceleration < 0, 0)
        
        acceleration_asymmetry = (pos_acceleration.rolling(window=10).mean() - 
                                neg_acceleration.abs().rolling(window=10).mean())
        
        return acceleration_asymmetry.fillna(0)
    
    def calculate_volume_deceleration(data, window=15):
        """Calculate volume deceleration patterns"""
        volume = data['volume']
        
        # Volume momentum
        volume_momentum = volume.pct_change(5).rolling(window=5).mean()
        
        # Volume acceleration
        volume_acceleration = volume_momentum.diff(3).rolling(window=5).mean()
        
        # Volume deceleration (negative acceleration)
        volume_deceleration = -volume_acceleration.where(volume_acceleration < 0, 0)
        
        return volume_deceleration.fillna(0)
    
    # Calculate acceleration-deceleration components
    price_acceleration = calculate_acceleration_profile(data)
    volume_deceleration = calculate_volume_deceleration(data)
    
    # Combine signals with non-linear scaling
    accel_decel_signal = (price_acceleration * volume_deceleration * 
                         np.exp(-0.1 * np.abs(price_acceleration)))
    
    # Multi-resolution Regime Persistence
    def calculate_regime_persistence(data, short_window=5, medium_window=20, long_window=50):
        """Calculate regime persistence across multiple timeframes"""
        close_prices = data['close']
        volume = data['volume']
        
        # Short-term regime (price-volume relationship)
        price_volume_corr = (close_prices.pct_change().rolling(window=short_window).corr(
            volume.pct_change())).fillna(0)
        
        # Medium-term regime (momentum and volatility)
        momentum = close_prices.pct_change(medium_window)
        volatility = close_prices.pct_change().rolling(window=medium_window).std()
        medium_regime = momentum / (volatility + 1e-10)
        
        # Long-term regime (trend persistence)
        long_trend = close_prices.rolling(window=long_window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == long_window else 0)
        
        # Combine regime signals
        regime_strength = (price_volume_corr.rolling(window=10).std() + 
                          medium_regime.rolling(window=10).std() + 
                          long_trend.rolling(window=10).std())
        
        return regime_strength.fillna(0)
    
    regime_persistence = calculate_regime_persistence(data)
    
    # CDF Break Detection
    def calculate_cdf_breaks(data, window=100):
        """Detect breaks in return distribution using CDF comparison"""
        returns = data['close'].pct_change().fillna(0)
        
        cdf_breaks = []
        for i in range(len(data)):
            if i < window * 2:
                cdf_breaks.append(0)
                continue
            
            # Historical CDF (first half of window)
            hist_returns = returns.iloc[i-window*2:i-window]
            # Current CDF (second half of window)
            current_returns = returns.iloc[i-window:i]
            
            if len(hist_returns) < 10 or len(current_returns) < 10:
                cdf_breaks.append(0)
                continue
            
            # Calculate KS statistic as break measure
            try:
                ks_stat = stats.ks_2samp(hist_returns, current_returns).statistic
                # Direction based on mean change
                direction = 1 if current_returns.mean() > hist_returns.mean() else -1
                cdf_breaks.append(ks_stat * direction)
            except:
                cdf_breaks.append(0)
        
        return pd.Series(cdf_breaks, index=data.index)
    
    cdf_break_signal = calculate_cdf_breaks(data)
    
    # Combine all signals with appropriate weights
    for i in range(len(data)):
        if i < 100:  # Warm-up period
            alpha_signal.iloc[i] = 0
            continue
        
        # Weighted combination of signals
        fractal_signal = fractal_roc.iloc[i] * 0.3
        accel_signal = accel_decel_signal.iloc[i] * 0.25
        regime_signal = regime_persistence.iloc[i] * 0.25
        cdf_signal = cdf_break_signal.iloc[i] * 0.2
        
        alpha_signal.iloc[i] = (fractal_signal + accel_signal + 
                               regime_signal + cdf_signal)
    
    # Final smoothing and normalization
    alpha_signal = alpha_signal.rolling(window=5).mean().fillna(0)
    
    # Ensure no look-ahead bias
    alpha_signal = alpha_signal.shift(1).fillna(0)
    
    return alpha_signal
