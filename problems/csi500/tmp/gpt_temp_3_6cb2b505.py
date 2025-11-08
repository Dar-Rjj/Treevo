import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics & Regime Transition Patterns alpha factor
    Combines multi-scale fractal analysis with regime detection for predictive signals
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Use close price for calculations
    close = data['close']
    volume = data['volume']
    high = data['high']
    low = data['low']
    
    # 1. Multi-Scale Price Fractal Analysis
    # Hurst Exponent Calculation using rescaled range analysis
    def hurst_exponent(series, max_lag=20):
        """Calculate Hurst exponent using R/S analysis"""
        lags = range(2, min(max_lag + 1, len(series) - 1))
        tau = []
        for lag in lags:
            # Create non-overlapping subseries
            subseries = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
            if len(subseries) < 2:
                continue
                
            rs_values = []
            for sub in subseries:
                if len(sub) < 2:
                    continue
                # Calculate mean-adjusted series
                mean_adj = sub - np.mean(sub)
                # Calculate cumulative deviations
                cumulative = np.cumsum(mean_adj)
                # Calculate range
                R = np.max(cumulative) - np.min(cumulative)
                # Calculate standard deviation
                S = np.std(sub, ddof=1)
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) < 2:
            return 0.5
        
        # Linear regression of log(tau) vs log(lag)
        lags_log = np.log(list(lags)[:len(tau)])
        slope, _, _, _, _ = linregress(lags_log, tau)
        return slope
    
    # Rolling Hurst exponent (21-day window)
    hurst_values = []
    for i in range(len(close)):
        if i < 40:  # Need enough data for calculation
            hurst_values.append(0.5)
        else:
            window_data = close.iloc[i-40:i+1]
            hurst_values.append(hurst_exponent(window_data))
    
    data['hurst'] = hurst_values
    
    # 2. Volume Fractal Geometry - Power Law Analysis
    def volume_power_law_exponent(volume_series, window=20):
        """Estimate power law exponent for volume distribution"""
        if len(volume_series) < window:
            return 1.0
        
        # Log-log rank-size analysis
        sorted_vol = np.sort(volume_series)[::-1]
        ranks = np.arange(1, len(sorted_vol) + 1)
        
        # Remove zeros and take logs
        mask = (sorted_vol > 0) & (ranks > 0)
        if np.sum(mask) < 3:
            return 1.0
            
        log_ranks = np.log(ranks[mask])
        log_volumes = np.log(sorted_vol[mask])
        
        # Linear regression for power law exponent
        slope, _, _, _, _ = linregress(log_ranks, log_volumes)
        return -slope  # Negative of slope gives power law exponent
    
    # Rolling volume power law exponent
    volume_exponents = []
    for i in range(len(volume)):
        if i < 40:
            volume_exponents.append(1.0)
        else:
            window_data = volume.iloc[i-20:i+1]
            volume_exponents.append(volume_power_law_exponent(window_data))
    
    data['volume_power_law'] = volume_exponents
    
    # 3. Regime Transition Detection - Early Warning Signals
    def early_warning_signals(series, window=15):
        """Calculate early warning signals for regime transitions"""
        if len(series) < window:
            return 0.0, 0.0, 0.0
        
        window_data = series.iloc[-window:]
        
        # Lag-1 autocorrelation
        acf1 = window_data.autocorr(lag=1) if len(window_data) > 1 else 0
        
        # Variance growth rate (rolling variance ratio)
        if len(series) >= 2 * window:
            var_short = series.iloc[-window:].var()
            var_long = series.iloc[-2*window:-window].var()
            var_ratio = var_short / var_long if var_long > 0 else 1.0
        else:
            var_ratio = 1.0
        
        # Skewness change
        skew_current = window_data.skew() if len(window_data) > 2 else 0
        
        return acf1, var_ratio, skew_current
    
    # Calculate EWS for price and volume
    ews_price = []
    ews_volume = []
    for i in range(len(close)):
        if i < 30:
            ews_price.append((0, 1, 0))
            ews_volume.append((0, 1, 0))
        else:
            price_window = close.iloc[max(0, i-29):i+1]
            volume_window = volume.iloc[max(0, i-29):i+1]
            ews_price.append(early_warning_signals(price_window))
            ews_volume.append(early_warning_signals(volume_window))
    
    data['price_acf1'] = [x[0] for x in ews_price]
    data['price_var_ratio'] = [x[1] for x in ews_price]
    data['price_skew'] = [x[2] for x in ews_price]
    data['volume_acf1'] = [x[0] for x in ews_volume]
    data['volume_var_ratio'] = [x[1] for x in ews_volume]
    
    # 4. Fractal-Momentum Integration
    # Multi-timeframe momentum alignment
    def multi_scale_momentum(price_series):
        """Calculate momentum across different timeframes"""
        if len(price_series) < 20:
            return 0.0
        
        # Short-term momentum (5 days)
        mom_short = (price_series.iloc[-1] / price_series.iloc[-5] - 1) if len(price_series) >= 5 else 0
        
        # Medium-term momentum (10 days)
        mom_medium = (price_series.iloc[-1] / price_series.iloc[-10] - 1) if len(price_series) >= 10 else 0
        
        # Long-term momentum (20 days)
        mom_long = (price_series.iloc[-1] / price_series.iloc[-20] - 1) if len(price_series) >= 20 else 0
        
        # Momentum alignment score (product of normalized momentums)
        mom_scores = [mom_short, mom_medium, mom_long]
        non_zero = [m for m in mom_scores if abs(m) > 1e-10]
        
        if len(non_zero) >= 2:
            # Geometric mean of absolute values with sign from product
            sign = np.sign(np.prod(non_zero))
            magnitude = np.exp(np.mean(np.log(np.abs(non_zero) + 1e-10)))
            return sign * magnitude
        else:
            return 0.0
    
    # Rolling multi-scale momentum
    momentum_scores = []
    for i in range(len(close)):
        if i < 20:
            momentum_scores.append(0.0)
        else:
            window_data = close.iloc[i-19:i+1]
            momentum_scores.append(multi_scale_momentum(window_data))
    
    data['multi_scale_mom'] = momentum_scores
    
    # 5. Composite Fractal-Regime Factor
    # Combine all components into final alpha factor
    def calculate_composite_factor(row):
        """Combine fractal and regime signals into composite factor"""
        
        # Base signals
        hurst = row['hurst']
        volume_pl = row['volume_power_law']
        mom_score = row['multi_scale_mom']
        
        # Regime transition signals (negative for mean reversion, positive for trending)
        regime_signal = (
            row['price_acf1'] +  # Higher autocorrelation suggests persistence
            (row['price_var_ratio'] - 1) +  # Increasing variance suggests instability
            abs(row['price_skew'])  # Higher skewness suggests regime change
        )
        
        # Volume regime signal
        volume_regime = row['volume_acf1'] + (row['volume_var_ratio'] - 1)
        
        # Fractal efficiency signal (Hurst away from 0.5 indicates predictability)
        fractal_efficiency = abs(hurst - 0.5)
        
        # Volume scaling efficiency (deviation from efficient market power law ~1.5)
        volume_efficiency = abs(volume_pl - 1.5)
        
        # Composite factor logic:
        # - High momentum + high fractal efficiency = strong trend following
        # - Low momentum + regime transitions = mean reversion opportunities
        # - Volume scaling anomalies provide additional confirmation
        
        if abs(mom_score) > 0.02:  # Significant momentum
            factor = mom_score * fractal_efficiency * (1 + volume_efficiency)
        else:  # Mean reversion regime
            factor = -regime_signal * volume_regime * fractal_efficiency
        
        return factor
    
    # Calculate final alpha factor
    alpha_factor = data.apply(calculate_composite_factor, axis=1)
    
    # Normalize using rolling z-score (21-day window)
    def rolling_zscore(series, window=21):
        """Calculate rolling z-score"""
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        return (series - rolling_mean) / (rolling_std + 1e-10)
    
    alpha_normalized = rolling_zscore(alpha_factor, window=21)
    
    return alpha_normalized
