import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Scale Fractal Dynamics alpha factor combining price-volume phase space analysis
    with multi-timeframe fractal characteristics for horizon-adaptive signals.
    """
    result = pd.Series(index=df.index, dtype=float)
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    for i in range(60, len(df)):  # Need sufficient history for calculations
        current_data = df.iloc[:i+1]
        
        # 1. Multi-timeframe Hurst Exponent Difference
        hurst_diff = _calculate_hurst_difference(current_data)
        
        # 2. Multi-fractal Spectrum Width Approximation
        spectrum_width = _estimate_spectrum_width(current_data)
        
        # 3. Phase Space Correlation Dimension
        corr_dim = _compute_correlation_dimension(current_data)
        
        # 4. Price-Volume Coupling Strength
        coupling_strength = _measure_price_volume_coupling(current_data)
        
        # 5. Multi-scale Information Flow
        info_flow = _estimate_information_flow(current_data)
        
        # 6. Fractal Residual Analysis
        residual_pattern = _analyze_fractal_residuals(current_data)
        
        # Combine signals with horizon-adaptive weighting
        # Intraday: Fractal Momentum × Attractor Compression
        intraday_signal = hurst_diff * (1.0 / (1.0 + np.abs(corr_dim - 2.0)))
        
        # 1-2 day: Transfer Entropy Direction
        short_term_signal = info_flow * spectrum_width
        
        # 3-5 day: Multi-fractal Spectrum Changes
        long_term_signal = residual_pattern * coupling_strength
        
        # Final alpha: weighted combination across horizons
        alpha_value = (0.4 * intraday_signal + 
                      0.35 * short_term_signal + 
                      0.25 * long_term_signal)
        
        result.iloc[i] = alpha_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result

def _calculate_hurst_difference(data):
    """Calculate Hurst exponent difference between short and long timeframes"""
    if len(data) < 60:
        return 0
    
    prices = data['close'].values
    
    # Short-term Hurst (approx 1-min equivalent)
    short_window = min(30, len(prices) // 2)
    hurst_short = _hurst_exponent(prices[-short_window:])
    
    # Long-term Hurst (approx 15-min equivalent)
    long_window = min(120, len(prices))
    hurst_long = _hurst_exponent(prices[-long_window:])
    
    return hurst_short - hurst_long

def _hurst_exponent(time_series):
    """Calculate Hurst exponent using R/S analysis"""
    if len(time_series) < 10:
        return 0.5
    
    lags = range(2, min(20, len(time_series) // 2))
    tau = []
    
    for lag in lags:
        # Calculate R/S for each lag
        rs_values = []
        for i in range(0, len(time_series) - lag, lag):
            segment = time_series[i:i + lag]
            if len(segment) < 2:
                continue
            mean_val = np.mean(segment)
            deviations = segment - mean_val
            z = np.cumsum(deviations)
            r = np.max(z) - np.min(z)
            s = np.std(segment)
            if s > 0:
                rs_values.append(r / s)
        
        if rs_values:
            tau.append(np.log(np.mean(rs_values)))
        else:
            tau.append(0)
    
    if len(tau) > 1:
        try:
            hurst, _ = linregress(np.log(lags[:len(tau)]), tau)[:2]
            return max(0.01, min(0.99, hurst))
        except:
            return 0.5
    
    return 0.5

def _estimate_spectrum_width(data):
    """Estimate multi-fractal spectrum width using price moments"""
    if len(data) < 50:
        return 0
    
    prices = data['close'].values[-50:]
    returns = np.diff(np.log(prices))
    
    if len(returns) < 10:
        return 0
    
    # Calculate moments for spectrum approximation
    moments = []
    for q in [0.5, 1.0, 1.5, 2.0, 2.5]:
        if q == 1:
            moment_val = np.mean(np.abs(returns))
        else:
            moment_val = np.mean(np.abs(returns) ** q) ** (1/q)
        moments.append(moment_val)
    
    spectrum_width = np.max(moments) - np.min(moments)
    return spectrum_width / (np.mean(moments) + 1e-8)

def _compute_correlation_dimension(data):
    """Compute correlation dimension from time-delay embedded price series"""
    if len(data) < 40:
        return 2.0
    
    prices = data['close'].values[-40:]
    
    # Time-delay embedding
    embedding_dim = 3
    tau = 5  # time delay
    
    points = []
    for i in range(len(prices) - (embedding_dim - 1) * tau):
        point = [prices[i + j * tau] for j in range(embedding_dim)]
        points.append(point)
    
    if len(points) < 10:
        return 2.0
    
    points = np.array(points)
    
    # Calculate correlation sum for different radii
    radii = np.logspace(-3, -1, 5)
    correlation_sums = []
    
    for r in radii:
        count = 0
        total_pairs = 0
        
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                distance = np.linalg.norm(points[i] - points[j])
                if distance < r:
                    count += 1
                total_pairs += 1
        
        if total_pairs > 0:
            correlation_sums.append(count / total_pairs)
        else:
            correlation_sums.append(0)
    
    # Fit slope in log-log space for correlation dimension
    valid_indices = [i for i, cs in enumerate(correlation_sums) if cs > 0]
    if len(valid_indices) >= 3:
        try:
            log_r = np.log(radii[valid_indices])
            log_c = np.log(correlation_sums[valid_indices])
            slope, _ = linregress(log_r, log_c)[:2]
            return max(0.5, min(3.0, slope))
        except:
            return 2.0
    
    return 2.0

def _measure_price_volume_coupling(data):
    """Measure price-volume coupling using mutual nearest neighbors ratio"""
    if len(data) < 30:
        return 0
    
    recent_data = data.iloc[-30:]
    prices = recent_data['close'].values
    volumes = recent_data['volume'].values
    
    # Normalize
    price_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-8)
    volume_norm = (volumes - np.mean(volumes)) / (np.std(volumes) + 1e-8)
    
    # Calculate correlation as coupling proxy
    if len(price_norm) > 5:
        correlation = np.corrcoef(price_norm, volume_norm)[0, 1]
        return np.abs(correlation) if not np.isnan(correlation) else 0
    
    return 0

def _estimate_information_flow(data):
    """Estimate information flow using multi-scale correlation"""
    if len(data) < 40:
        return 0
    
    prices = data['close'].values
    
    # Fine scale (1-min equivalent)
    fine_scale = prices[-20:]
    fine_returns = np.diff(np.log(fine_scale))
    
    # Coarse scale (5-min equivalent)
    coarse_scale = prices[-40::2]  # decimate
    if len(coarse_scale) > 10:
        coarse_returns = np.diff(np.log(coarse_scale))
        
        # Information flow as cross-correlation
        min_len = min(len(fine_returns), len(coarse_returns))
        if min_len > 5:
            fine_aligned = fine_returns[-min_len:]
            coarse_aligned = coarse_returns[-min_len:]
            
            # Lead-lag correlation
            correlation = np.corrcoef(fine_aligned, coarse_aligned)[0, 1]
            return correlation if not np.isnan(correlation) else 0
    
    return 0

def _analyze_fractal_residuals(data):
    """Analyze fractal Brownian motion fit residuals"""
    if len(data) < 30:
        return 0
    
    prices = data['close'].values[-30:]
    log_prices = np.log(prices)
    
    # Simple linear trend as FBM approximation
    x = np.arange(len(log_prices))
    try:
        slope, intercept, _, _, _ = linregress(x, log_prices)
        trend = intercept + slope * x
        residuals = log_prices - trend
        
        # Analyze residual autocorrelation pattern
        if len(residuals) > 5:
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
            return -autocorr if not np.isnan(autocorr) else 0  # Negative for mean reversion signal
    except:
        pass
    
    return 0
