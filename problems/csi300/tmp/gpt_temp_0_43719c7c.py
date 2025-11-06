import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Scale Fractal Dynamics Alpha Factor
    Combines fractal dimension analysis, phase space reconstruction, and multi-scale information cascade
    to generate predictive signals across different time horizons.
    """
    
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate returns for different time scales
    data['ret_1'] = data['close'].pct_change(1)
    data['ret_5'] = data['close'].pct_change(5)
    data['ret_15'] = data['close'].pct_change(15)
    
    # 1. Fractal Dimension Analysis Components
    def hurst_exponent(series, max_lag=20):
        """Calculate Hurst exponent using rescaled range analysis"""
        if len(series) < max_lag * 2:
            return 0.5
        
        lags = range(2, max_lag)
        tau = []
        for lag in lags:
            # Create non-overlapping windows
            if len(series) < lag * 2:
                continue
                
            # Calculate R/S for each window
            rs_values = []
            for i in range(0, len(series) - lag, lag):
                window = series.iloc[i:i+lag]
                if len(window) < 2:
                    continue
                    
                # Detrend the window
                mean_val = window.mean()
                deviations = window - mean_val
                cumulative = deviations.cumsum()
                
                # Calculate R/S
                R = cumulative.max() - cumulative.min()
                S = window.std()
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
            else:
                tau.append(0)
        
        if len(tau) > 2:
            # Fit log-log plot
            lags_log = np.log(lags[:len(tau)])
            slope, _, _, _, _ = linregress(lags_log, tau)
            return slope
        return 0.5
    
    # Calculate multi-scale Hurst exponents
    data['hurst_5'] = data['close'].rolling(100).apply(
        lambda x: hurst_exponent(x, max_lag=10), raw=False
    )
    data['hurst_15'] = data['close'].rolling(200).apply(
        lambda x: hurst_exponent(x, max_lag=15), raw=False
    )
    
    # 2. Phase Space Reconstruction Components
    def correlation_dimension(series, embedding_dim=3, delay=1, max_radius=0.1):
        """Estimate correlation dimension using Grassberger-Procaccia algorithm"""
        if len(series) < embedding_dim * 2:
            return 1.0
            
        # Create time-delay embedding
        embedded = []
        for i in range(len(series) - (embedding_dim-1)*delay):
            point = [series.iloc[i + j*delay] for j in range(embedding_dim)]
            embedded.append(point)
        
        if len(embedded) < 10:
            return 1.0
            
        embedded = np.array(embedded)
        
        # Calculate correlation sum for different radii
        radii = np.linspace(0.01, max_radius, 10)
        C_r = []
        
        for r in radii:
            if r <= 0:
                continue
            count = 0
            total_pairs = 0
            
            for i in range(len(embedded)):
                for j in range(i+1, len(embedded)):
                    distance = np.linalg.norm(embedded[i] - embedded[j])
                    if distance < r:
                        count += 1
                    total_pairs += 1
            
            if total_pairs > 0:
                C_r.append(count / total_pairs)
            else:
                C_r.append(0)
        
        # Fit log-log to get correlation dimension
        valid_idx = [i for i, (r, c) in enumerate(zip(radii, C_r)) 
                    if r > 0 and c > 0 and not np.isinf(np.log(c))]
        
        if len(valid_idx) > 3:
            log_r = np.log([radii[i] for i in valid_idx])
            log_C = np.log([C_r[i] for i in valid_idx])
            slope, _, _, _, _ = linregress(log_r, log_C)
            return abs(slope)
        
        return 1.0
    
    # Calculate correlation dimension
    data['corr_dim'] = data['close'].rolling(150).apply(
        lambda x: correlation_dimension(x), raw=False
    )
    
    # 3. Multi-scale Information Cascade Components
    def transfer_entropy_estimate(x, y, delay=1):
        """Simple estimate of information transfer between two series"""
        if len(x) < 20:
            return 0
            
        # Use correlation of past values with future values as proxy
        x_past = x[:-delay]
        y_future = y[delay:]
        
        if len(x_past) != len(y_future) or len(x_past) < 10:
            return 0
            
        corr = np.corrcoef(x_past, y_future)[0, 1]
        if np.isnan(corr):
            return 0
            
        return abs(corr)
    
    # Calculate multi-scale transfer entropy
    data['te_1_5'] = data['close'].rolling(100).apply(
        lambda x: transfer_entropy_estimate(
            x.diff().iloc[:-5], x.diff().iloc[5:], delay=5
        ) if len(x) >= 30 else 0, raw=False
    )
    
    # 4. Fractal Momentum Divergence
    data['fractal_momentum'] = (
        data['hurst_5'].rolling(10).mean() - data['hurst_15'].rolling(20).mean()
    )
    
    # 5. Attractor Compression Factor
    data['attractor_compression'] = data['corr_dim'].pct_change(5)
    
    # 6. Multi-scale Information Cascade Efficiency
    data['cascade_efficiency'] = (
        data['te_1_5'].rolling(10).mean() * 
        (1 - data['volume'].pct_change(5).abs().rolling(10).mean())
    )
    
    # 7. Horizon-Adaptive Alpha Integration
    # Ultra-short term component (intraday)
    ultra_short = (
        data['fractal_momentum'].fillna(0) * 
        (-data['attractor_compression'].fillna(0))
    )
    
    # Short-term component (1-2 days)
    short_term = data['te_1_5'].fillna(0)
    
    # Medium-term component (3-5 days)
    medium_term = data['hurst_5'].pct_change(10).fillna(0)
    
    # Fractal stability regime detection
    hurst_stability = data['hurst_5'].rolling(20).std().fillna(0.1)
    regime_weight = 1 / (1 + hurst_stability * 10)
    
    # Final alpha: regime-dependent combination
    alpha = (
        regime_weight * ultra_short.rolling(5).mean() +
        (1 - regime_weight) * short_term.rolling(3).mean() +
        medium_term.rolling(8).mean()
    )
    
    # Normalize and clean
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = (alpha - alpha.rolling(252).mean()) / alpha.rolling(252).std()
    alpha = alpha.fillna(0)
    
    return alpha
