import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime-Based Alpha with Fractal Market Structure
    """
    data = df.copy()
    
    # Market Regime Identification
    # Price Efficiency Ratio: Absolute 5-day return divided by 20-day volatility
    returns_5d = data['close'].pct_change(5).abs()
    vol_20d = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    efficiency_ratio = returns_5d / (vol_20d * np.sqrt(5))
    
    # Hurst Exponent Proxy using rescaled range
    def hurst_proxy(series, window=10):
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                hurst_values.append(np.nan)
                continue
                
            # Calculate mean and cumulative deviations
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative = deviations.cumsum()
            
            # Range and standard deviation
            R = cumulative.max() - cumulative.min()
            S = window_data.std()
            
            if S > 0:
                # Proxy for Hurst: log(R/S) / log(window)
                hurst_proxy = np.log(R / S) / np.log(window)
                hurst_values.append(hurst_proxy)
            else:
                hurst_values.append(np.nan)
        
        return pd.Series(hurst_values, index=series.index)
    
    hurst = hurst_proxy(data['close'], window=10)
    
    # Regime Classification
    trending_regime = ((efficiency_ratio > efficiency_ratio.rolling(50).quantile(0.7)) & 
                      (hurst > hurst.rolling(50).quantile(0.7)))
    mean_reverting_regime = ((efficiency_ratio < efficiency_ratio.rolling(50).quantile(0.3)) & 
                           (hurst < hurst.rolling(50).quantile(0.3)))
    transition_regime = (~trending_regime & ~mean_reverting_regime)
    
    # Fractal Price Structure Analysis
    # Multi-scale local extrema detection
    def find_local_extrema(series, window):
        highs = series.rolling(window=window, center=True).max()
        lows = series.rolling(window=window, center=True).min()
        high_peaks = (series == highs) & (series > series.shift(1)) & (series > series.shift(-1))
        low_troughs = (series == lows) & (series < series.shift(1)) & (series < series.shift(-1))
        return high_peaks, low_troughs
    
    high_3d, low_3d = find_local_extrema(data['high'], 3)
    high_7d, low_7d = find_local_extrema(data['high'], 7)
    high_15d, low_15d = find_local_extrema(data['high'], 15)
    
    # Price Fractal Dimension Proxy
    def price_fractal_dimension(close_prices, scales=[3, 7, 15]):
        fractal_scores = []
        for i in range(len(close_prices)):
            if i < max(scales):
                fractal_scores.append(np.nan)
                continue
                
            level_counts = []
            current_price = close_prices.iloc[i]
            
            for scale in scales:
                window_data = close_prices.iloc[i-scale+1:i+1]
                price_levels = pd.cut(window_data, bins=10, labels=False)
                unique_levels = len(set(price_levels.dropna()))
                level_counts.append(unique_levels)
            
            # Simple fractal dimension proxy: scaling of level counts
            if len(level_counts) > 1:
                log_scales = np.log(scales)
                log_counts = np.log(level_counts)
                fractal_dim = np.polyfit(log_scales, log_counts, 1)[0]
                fractal_scores.append(fractal_dim)
            else:
                fractal_scores.append(np.nan)
        
        return pd.Series(fractal_scores, index=close_prices.index)
    
    price_fractal = price_fractal_dimension(data['close'])
    
    # Volume Fractal Analysis
    def volume_fractal_analysis(volume, scales=[5, 10, 20]):
        fractal_scores = []
        for i in range(len(volume)):
            if i < max(scales):
                fractal_scores.append(np.nan)
                continue
                
            volume_clusters = []
            for scale in scales:
                window_vol = volume.iloc[i-scale+1:i+1]
                vol_percentiles = window_vol.rank(pct=True)
                # Measure clustering: variance of percentile distribution
                cluster_coef = 1 - vol_percentiles.std()
                volume_clusters.append(cluster_coef)
            
            fractal_dim = np.mean(volume_clusters)
            fractal_scores.append(fractal_dim)
        
        return pd.Series(fractal_scores, index=volume.index)
    
    volume_fractal = volume_fractal_analysis(data['volume'])
    
    # Volume-Price Fractal Alignment
    fractal_alignment = (price_fractal.rolling(10).corr(volume_fractal)).fillna(0)
    
    # Regime-Adaptive Signal Generation
    # Trending Regime Signals
    momentum_5d = data['close'].pct_change(5)
    momentum_accel = momentum_5d.diff(3)  # Rate of change of momentum
    
    # Breakout signals relative to multi-scale resistance
    resistance_levels = []
    for i in range(len(data)):
        current_high = data['high'].iloc[i]
        # Look for recent resistance levels
        lookback = min(20, i)
        if lookback > 0:
            recent_highs = data['high'].iloc[i-lookback:i]
            resistance = recent_highs.quantile(0.8)
            breakout_strength = (current_high - resistance) / resistance
            resistance_levels.append(breakout_strength)
        else:
            resistance_levels.append(0)
    
    breakout_signal = pd.Series(resistance_levels, index=data.index)
    
    # Mean-Reverting Regime Signals
    def overextension_signal(close, high, low, scales=[3, 7, 15]):
        signals = []
        for i in range(len(close)):
            if i < max(scales):
                signals.append(np.nan)
                continue
                
            current_close = close.iloc[i]
            overextension_scores = []
            
            for scale in scales:
                window_high = high.iloc[i-scale+1:i+1].max()
                window_low = low.iloc[i-scale+1:i+1].min()
                window_range = window_high - window_low
                
                if window_range > 0:
                    # Position within recent range
                    position = (current_close - window_low) / window_range
                    # Overextension: far from middle (0.5)
                    overextension = abs(position - 0.5) * 2
                    overextension_scores.append(overextension)
                else:
                    overextension_scores.append(0)
            
            signals.append(np.mean(overextension_scores))
        
        return pd.Series(signals, index=close.index)
    
    overextension = overextension_signal(data['close'], data['high'], data['low'])
    
    # Volume compression at extremes
    volume_compression = (data['volume'] / data['volume'].rolling(20).mean()).rolling(5).std()
    
    # Dynamic Risk Adjustment
    # Regime-specific volatility
    trending_vol = momentum_5d.rolling(10).std()
    mean_rev_vol = (data['high'] - data['low']).rolling(10).mean() / data['close'].rolling(10).mean()
    
    regime_vol = pd.Series(np.nan, index=data.index)
    regime_vol[trending_regime] = trending_vol[trending_regime]
    regime_vol[mean_reverting_regime] = mean_rev_vol[mean_reverting_regime]
    regime_vol[transition_regime] = (trending_vol + mean_rev_vol)[transition_regime] / 2
    
    # Fractal-based position sizing
    position_size = 1 / (1 + price_fractal.rolling(10).mean())
    
    # Adaptive Factor Integration
    # Regime-weighted signal combination
    trending_weight = trending_regime.astype(float)
    mean_rev_weight = mean_reverting_regime.astype(float)
    transition_weight = transition_regime.astype(float)
    
    # Normalize weights
    total_weight = trending_weight + mean_rev_weight + transition_weight
    trending_weight = trending_weight / total_weight.replace(0, 1)
    mean_rev_weight = mean_rev_weight / total_weight.replace(0, 1)
    transition_weight = transition_weight / total_weight.replace(0, 1)
    
    # Trending signals: momentum and breakout
    trending_signal = (momentum_accel * 0.6 + breakout_signal * 0.4)
    
    # Mean-reverting signals: overextension and volume compression
    mean_rev_signal = (overextension * 0.7 - volume_compression * 0.3)
    
    # Transition signals: blend of both
    transition_signal = (trending_signal + mean_rev_signal) / 2
    
    # Combine signals with regime weights
    combined_signal = (trending_weight * trending_signal + 
                      mean_rev_weight * mean_rev_signal + 
                      transition_weight * transition_signal)
    
    # Fractal structure validation
    fractal_validated = combined_signal * (1 + fractal_alignment * 0.3)
    
    # Final factor with risk adjustment
    final_factor = fractal_validated * position_size / (regime_vol.replace(0, 1))
    
    return final_factor.fillna(0)
