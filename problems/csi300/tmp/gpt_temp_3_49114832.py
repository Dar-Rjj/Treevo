import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Efficiency Momentum with Volume-Pressure Integration
    Combines multi-scale fractal analysis, efficiency-adjusted momentum, 
    pressure-regime volume clustering, and amplitude-aware mean reversion
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Scale Fractal Analysis
    # Price Fractal Dimension Calculation using Hurst exponent approximation
    def hurst_exponent(series, window=10):
        """Calculate Hurst exponent approximation for fractal dimension"""
        lags = range(2, min(6, window))
        tau = []
        for lag in lags:
            # RS method for Hurst exponent
            series_lag = series.diff(lag).dropna()
            if len(series_lag) > 0:
                R = series_lag.max() - series_lag.min()
                S = series_lag.std()
                if S > 0:
                    tau.append(np.log(R/S) / np.log(lag))
        return np.mean(tau) if tau else 0.5
    
    # Calculate price fractal dimension
    data['hurst_price'] = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: hurst_exponent(x), raw=False
    )
    
    # Volume fractal synchronization
    def volume_fractal(volume_series, window=10):
        """Calculate volume clustering fractal dimension"""
        volume_changes = volume_series.pct_change().abs()
        return volume_changes.rolling(window=window).std() / (volume_changes.rolling(window=window).mean() + 1e-8)
    
    data['volume_fractal'] = volume_fractal(data['volume'], window=10)
    
    # 2. Efficiency-Adjusted Momentum
    # Daily efficiency ratio
    data['efficiency_ratio'] = (data['close'] - data['open']).abs() / (data['high'] - data['low'] + 1e-8)
    data['volume_weighted_efficiency'] = data['efficiency_ratio'] * (data['volume'] / data['volume'].rolling(20).mean())
    
    # Efficiency momentum score
    data['efficiency_momentum'] = data['volume_weighted_efficiency'].rolling(5).mean() - data['volume_weighted_efficiency'].rolling(20).mean()
    
    # 3. Pressure-Regime Volume Clustering
    # Daily pressure index
    data['pressure_index'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['cumulative_pressure'] = data['pressure_index'].rolling(5).sum()
    
    # Volume regime identification
    data['volume_percentile'] = data['volume'].rolling(20).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8), raw=True
    )
    
    # Pressure release detection
    data['pressure_change'] = data['pressure_index'].pct_change(periods=1)
    data['pressure_reversal'] = (data['pressure_change'] * data['pressure_change'].shift(1) < 0).astype(int)
    
    # 4. Amplitude-Aware Mean Reversion
    # Contextual deviation analysis
    data['price_median'] = data['close'].rolling(window=10).median()
    data['price_deviation'] = (data['close'] - data['price_median']) / data['price_median']
    
    # Amplitude contextualization
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_daily_range'] = data['daily_range'].rolling(window=10).mean()
    data['volatility_regime'] = (data['daily_range'] > data['avg_daily_range']).astype(int)
    
    # Adaptive reversion bands
    data['reversion_band'] = data['avg_daily_range'] * (1 + 0.5 * data['volatility_regime'])
    data['extreme_deviation'] = (data['price_deviation'].abs() > data['reversion_band']).astype(int)
    
    # Volume-enhanced reversion signals
    data['volume_percentile_reversion'] = data['volume_percentile'] * data['extreme_deviation'] * np.sign(-data['price_deviation'])
    
    # 5. Composite Alpha Generation
    # Fractal-efficiency integration
    data['fractal_efficiency_score'] = (
        data['hurst_price'] * data['efficiency_momentum'] * 
        np.where(data['volume_fractal'] > data['volume_fractal'].rolling(20).mean(), 1, 0.5)
    )
    
    # Regime-transition signals
    data['regime_transition'] = (
        data['pressure_reversal'] * data['volume_percentile'] * 
        np.sign(data['pressure_change'])
    )
    
    # Final composite alpha
    alpha = (
        0.4 * data['fractal_efficiency_score'] +
        0.3 * data['volume_percentile_reversion'] +
        0.2 * data['regime_transition'] +
        0.1 * data['efficiency_momentum']
    )
    
    # Normalize and clean
    alpha = (alpha - alpha.rolling(50).mean()) / (alpha.rolling(50).std() + 1e-8)
    alpha = alpha.replace([np.inf, -np.inf], 0).fillna(0)
    
    return alpha
