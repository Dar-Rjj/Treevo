import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal-Based Alpha Factor combining multi-scale fractal analysis, regime detection,
    information asymmetry, and structural memory patterns.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Scale Fractal Analysis
    # Hurst Exponent calculation (5, 10, 20 days)
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using R/S analysis"""
        lags = range(2, min(window, len(series)))
        tau = []
        for lag in lags:
            # Create non-overlapping subseries
            subseries = [series[i:i+lag] for i in range(0, len(series)-lag, lag)]
            if len(subseries) < 2:
                continue
                
            # Calculate R/S for each subseries
            rs_values = []
            for sub in subseries:
                if len(sub) < 2:
                    continue
                mean_sub = np.mean(sub)
                deviations = sub - mean_sub
                z = np.cumsum(deviations)
                r = np.max(z) - np.min(z)
                s = np.std(sub)
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) > 1:
            lags_log = np.log(lags[:len(tau)])
            hurst = np.polyfit(lags_log, tau, 1)[0]
            return hurst
        return np.nan
    
    # Calculate Hurst exponents for different windows
    for window in [5, 10, 20]:
        data[f'hurst_{window}'] = data['close'].rolling(window=window*2, min_periods=window*2).apply(
            lambda x: hurst_exponent(x, window), raw=False
        )
    
    # High-Low Range Fractal Spectrum
    data['range_fractal'] = (data['high'] - data['low']) / data['close']
    data['range_volatility'] = data['range_fractal'].rolling(window=10).std()
    
    # Volume Fractal Patterns
    data['volume_clustering'] = data['volume'].rolling(window=10).apply(
        lambda x: np.std(x) / (np.mean(x) + 1e-8), raw=False
    )
    
    # 2. Regime Transition Detection
    # Fractal Boundary Crossings
    data['hurst_5_ma'] = data['hurst_5'].rolling(window=5).mean()
    data['hurst_10_ma'] = data['hurst_10'].rolling(window=5).mean()
    
    # Dimension threshold breaks
    data['fractal_boundary'] = ((data['hurst_5'] > 0.7) & (data['hurst_5_ma'].shift(1) <= 0.7)).astype(int) - \
                              ((data['hurst_5'] < 0.3) & (data['hurst_5_ma'].shift(1) >= 0.3)).astype(int)
    
    # Fractal Compression Patterns
    data['hurst_volatility'] = data['hurst_5'].rolling(window=10).std()
    data['fractal_compression'] = (data['hurst_volatility'] < data['hurst_volatility'].rolling(window=20).quantile(0.2)).astype(int)
    
    # 3. Information Asymmetry
    # Price-Volume Decoupling
    price_returns = data['close'].pct_change()
    volume_returns = data['volume'].pct_change()
    
    data['price_volume_corr'] = price_returns.rolling(window=10).corr(volume_returns)
    data['fractal_divergence'] = (data['hurst_5'] - data['hurst_10']).abs() * (1 - data['price_volume_corr'].abs())
    
    # Price Movement Entropy
    def price_entropy(returns):
        """Calculate entropy of price movements"""
        if len(returns) < 5:
            return np.nan
        # Bin returns into 5 categories
        bins = np.percentile(returns.dropna(), [0, 20, 40, 60, 80, 100])
        hist, _ = np.histogram(returns, bins=bins)
        prob = hist / len(returns)
        prob = prob[prob > 0]
        entropy = -np.sum(prob * np.log(prob))
        return entropy
    
    data['price_entropy'] = price_returns.rolling(window=20).apply(price_entropy, raw=False)
    
    # 4. Structural Memory Analysis
    # Long-range dependence measure
    def autocorrelation_memory(series, max_lag=5):
        """Measure autocorrelation memory decay"""
        if len(series) < max_lag + 1:
            return np.nan
        autocorrs = [series.autocorr(lag=i) for i in range(1, max_lag+1)]
        autocorrs = [ac for ac in autocorrs if not pd.isna(ac)]
        if len(autocorrs) > 1:
            return np.mean(np.abs(autocorrs))
        return np.nan
    
    data['memory_decay'] = price_returns.rolling(window=20).apply(
        lambda x: autocorrelation_memory(x, 5), raw=False
    )
    
    # Fractal Stability
    data['fractal_stability'] = 1 / (1 + data['hurst_volatility'])
    
    # 5. Alpha Signal Generation
    # Fractal Timing Signals
    data['dimension_convergence'] = (data['hurst_5'] - data['hurst_10']).abs() * \
                                   (data['hurst_10'] - data['hurst_20']).abs()
    
    # Information Momentum
    data['directional_accumulation'] = (price_returns * data['volume']).rolling(window=5).sum()
    
    # Memory-Driven Prediction
    data['regime_probability'] = data['hurst_5'].rolling(window=10).apply(
        lambda x: np.mean((x > 0.5).astype(int)), raw=False
    )
    
    # Signal Integration with regime-adaptive weighting
    regime_weight = np.where(data['hurst_5'] > 0.6, 1.5, 
                           np.where(data['hurst_5'] < 0.4, 0.8, 1.0))
    
    # Final alpha factor combining all components
    alpha = (
        regime_weight * 
        (data['fractal_boundary'] * 0.3 + 
         data['fractal_divergence'] * 0.25 + 
         data['directional_accumulation'] * 0.2 + 
         data['memory_decay'] * 0.15 + 
         data['fractal_stability'] * 0.1)
    )
    
    # Normalize the alpha factor
    alpha_normalized = (alpha - alpha.rolling(window=20).mean()) / (alpha.rolling(window=20).std() + 1e-8)
    
    return alpha_normalized
