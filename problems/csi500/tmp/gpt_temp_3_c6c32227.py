import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics & Regime Transition Patterns alpha factor
    Combines multi-scale fractal analysis with regime detection for predictive signals
    """
    data = df.copy()
    
    # Multi-Scale Price Fractal Analysis
    def hurst_exponent(series, max_lag=20):
        """Calculate Hurst exponent using rescaled range analysis"""
        lags = range(2, max_lag + 1)
        tau = []
        for lag in lags:
            # Split series into non-overlapping windows
            n_windows = len(series) // lag
            if n_windows < 2:
                continue
                
            rs_values = []
            for i in range(n_windows):
                window = series[i*lag:(i+1)*lag]
                if len(window) < 2:
                    continue
                    
                # Calculate mean and deviations
                mean_val = window.mean()
                deviations = window - mean_val
                
                # Calculate cumulative deviations and range
                cumulative_deviations = deviations.cumsum()
                data_range = cumulative_deviations.max() - cumulative_deviations.min()
                
                # Calculate standard deviation
                std_dev = window.std()
                if std_dev > 0:
                    rs_values.append(data_range / std_dev)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            lags_log = np.log(lags[:len(tau)])
            hurst = stats.linregress(lags_log, tau)[0]
            return hurst
        return 0.5
    
    # Fractal Dimension Estimation
    def fractal_dimension(series, window=50):
        """Estimate fractal dimension using box counting method"""
        if len(series) < window:
            return 1.5
            
        window_data = series[-window:]
        normalized = (window_data - window_data.min()) / (window_data.max() - window_data.min() + 1e-8)
        
        # Simple box counting approximation
        n_boxes = 10
        counts = []
        box_sizes = [2**i for i in range(1, 6)]
        
        for size in box_sizes:
            if len(normalized) < size:
                continue
            count = 0
            for i in range(0, len(normalized) - size + 1, size):
                segment = normalized[i:i+size]
                if len(segment) > 0:
                    count += 1
            counts.append(count)
        
        if len(counts) > 1:
            box_log = np.log(box_sizes[:len(counts)])
            count_log = np.log(counts)
            slope = -stats.linregress(box_log, count_log)[0]
            return max(1.0, min(2.0, slope))
        return 1.5
    
    # Scale-Invariant Volatility Patterns
    def multi_scale_volatility(close_prices, windows=[5, 10, 20]):
        """Calculate volatility clustering across multiple timeframes"""
        volatilities = []
        for window in windows:
            if len(close_prices) >= window:
                returns = close_prices.pct_change().dropna()
                if len(returns) >= window:
                    vol = returns.rolling(window=window).std().iloc[-1]
                    volatilities.append(vol)
        
        if volatilities:
            # Measure volatility clustering through autocorrelation of volatilities
            vol_series = pd.Series(volatilities)
            if len(vol_series) > 1:
                clustering = vol_series.autocorr(lag=1)
                return clustering if not np.isnan(clustering) else 0
        return 0
    
    # Volume Fractal Geometry
    def volume_fractal_analysis(volume_series, price_series, window=30):
        """Analyze volume scaling properties and price-volume fractal correlation"""
        if len(volume_series) < window:
            return 0
            
        vol_data = volume_series[-window:]
        price_data = price_series[-window:]
        
        # Volume power law approximation
        vol_log = np.log(vol_data + 1)
        rank_log = np.log(np.arange(1, len(vol_data) + 1))
        
        if len(vol_log) > 2:
            power_law_slope = -stats.linregress(rank_log, vol_log.sort_values(ascending=False))[0]
        else:
            power_law_slope = 1.0
        
        # Volume-price fractal correlation
        vol_normalized = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min() + 1e-8)
        price_normalized = (price_data - price_data.min()) / (price_data.max() - price_data.min() + 1e-8)
        
        # Joint multi-fractal analysis through correlation of fluctuations
        vol_fluctuations = vol_normalized.diff().dropna()
        price_fluctuations = price_normalized.diff().dropna()
        
        if len(vol_fluctuations) > 1 and len(price_fluctuations) > 1:
            min_len = min(len(vol_fluctuations), len(price_fluctuations))
            fractal_corr = np.corrcoef(vol_fluctuations.iloc[:min_len], 
                                     price_fluctuations.iloc[:min_len])[0, 1]
            fractal_corr = 0 if np.isnan(fractal_corr) else fractal_corr
        else:
            fractal_corr = 0
        
        return power_law_slope * (1 + fractal_corr)
    
    # Regime Transition Detection
    def regime_transition_metrics(close_prices, volume, window=60):
        """Detect critical points and regime persistence"""
        if len(close_prices) < window:
            return 0, 0.5
            
        prices = close_prices[-window:]
        volumes = volume[-window:]
        
        # Early warning signals through variance and autocorrelation changes
        half_window = window // 2
        first_half_var = prices[:half_window].var()
        second_half_var = prices[half_window:].var()
        
        first_half_acf = prices[:half_window].autocorr(lag=1) or 0
        second_half_acf = prices[half_window:].autocorr(lag=1) or 0
        
        # Critical point identification
        var_ratio = second_half_var / (first_half_var + 1e-8)
        acf_change = second_half_acf - first_half_acf
        
        # Regime persistence through memory length
        returns = prices.pct_change().dropna()
        if len(returns) > 5:
            memory_length = 0
            for lag in range(1, min(10, len(returns))):
                acf = returns.autocorr(lag=lag) or 0
                if abs(acf) > 0.1:
                    memory_length += 1
                else:
                    break
        else:
            memory_length = 0
        
        critical_signal = var_ratio * (1 + abs(acf_change))
        persistence_metric = memory_length / 10.0
        
        return critical_signal, persistence_metric
    
    # Fractal-Momentum Integration
    def fractal_momentum_integration(close_prices, hurst, fractal_dim, window=20):
        """Combine fractal analysis with momentum signals"""
        if len(close_prices) < window:
            return 0
            
        # Multi-timeframe momentum alignment
        short_momentum = close_prices.pct_change(periods=5).iloc[-1] if len(close_prices) >= 6 else 0
        medium_momentum = close_prices.pct_change(periods=10).iloc[-1] if len(close_prices) >= 11 else 0
        long_momentum = close_prices.pct_change(periods=20).iloc[-1] if len(close_prices) >= 21 else 0
        
        momentum_alignment = (short_momentum * medium_momentum * long_momentum >= 0)
        
        # Fractal momentum persistence
        trend_persistence = hurst - 0.5  # Positive for trending, negative for mean-reverting
        
        # Regime-adaptive fractal factors
        if fractal_dim > 1.8:  # High fractal dimension - complex behavior
            regime_factor = 1.0 / (1.0 + abs(trend_persistence))
        elif fractal_dim < 1.2:  # Low fractal dimension - smooth behavior
            regime_factor = 1.0 + abs(trend_persistence)
        else:  # Medium fractal dimension
            regime_factor = 1.0
        
        momentum_score = (short_momentum + medium_momentum + long_momentum) / 3.0
        fractal_momentum = momentum_score * regime_factor * (2 if momentum_alignment else 1)
        
        return fractal_momentum
    
    # Main factor calculation
    factor_values = []
    
    for i in range(len(data)):
        if i < 60:  # Need sufficient history
            factor_values.append(0)
            continue
            
        current_data = data.iloc[:i+1]
        close_prices = current_data['close']
        volume = current_data['volume']
        
        # Calculate fractal components
        hurst = hurst_exponent(close_prices.tail(100))
        fractal_dim = fractal_dimension(close_prices.tail(50))
        vol_clustering = multi_scale_volatility(close_prices.tail(100))
        volume_fractal = volume_fractal_analysis(volume.tail(30), close_prices.tail(30))
        critical_signal, persistence_metric = regime_transition_metrics(close_prices.tail(60), volume.tail(60))
        fractal_momentum = fractal_momentum_integration(close_prices.tail(20), hurst, fractal_dim)
        
        # Combine components into final factor
        fractal_stability = (2.0 - fractal_dim) * (1.0 + abs(hurst - 0.5))
        regime_awareness = critical_signal * (1.0 - persistence_metric)
        volume_confirmation = volume_fractal * (1.0 + vol_clustering)
        
        # Final alpha factor
        alpha_factor = (fractal_momentum * fractal_stability + 
                       regime_awareness * volume_confirmation)
        
        factor_values.append(alpha_factor)
    
    return pd.Series(factor_values, index=data.index, name='fractal_regime_alpha')
