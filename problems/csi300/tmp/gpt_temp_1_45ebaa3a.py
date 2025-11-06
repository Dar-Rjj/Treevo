import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Momentum-Volume Synchronization factor
    Combines multi-scale momentum with fractal analysis and volume synchronization
    """
    data = df.copy()
    
    # Multi-Scale Momentum Assessment
    # Price Momentum (3,10,20-day returns)
    data['mom_3'] = data['close'].pct_change(3)
    data['mom_10'] = data['close'].pct_change(10)
    data['mom_20'] = data['close'].pct_change(20)
    
    # Volume Momentum (3,10,20-day growth)
    data['vol_mom_3'] = data['volume'].pct_change(3)
    data['vol_mom_10'] = data['volume'].pct_change(10)
    data['vol_mom_20'] = data['volume'].pct_change(20)
    
    # Fractal Market Analysis
    # Volatility Scaling (5/20, 20/60-day ratios)
    vol_5 = data['close'].pct_change().rolling(5).std()
    vol_20 = data['close'].pct_change().rolling(20).std()
    vol_60 = data['close'].pct_change().rolling(60).std()
    
    data['vol_ratio_5_20'] = vol_5 / vol_20
    data['vol_ratio_20_60'] = vol_20 / vol_60
    
    # Price Fractal Dimension (simplified Hurst exponent)
    def hurst_exponent(series, max_lag=20):
        lags = range(2, max_lag + 1)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    data['hurst_20'] = data['close'].rolling(100).apply(
        lambda x: hurst_exponent(x.values) if len(x.dropna()) >= 100 else np.nan, 
        raw=False
    )
    
    # Nonlinear Momentum Dynamics
    # Asymmetric Momentum (upside vs downside)
    upside_returns = data['close'].pct_change().clip(lower=0)
    downside_returns = -data['close'].pct_change().clip(upper=0)
    
    data['upside_mom'] = upside_returns.rolling(10).mean()
    data['downside_mom'] = downside_returns.rolling(10).mean()
    data['asym_mom_ratio'] = data['upside_mom'] / (data['downside_mom'] + 1e-8)
    
    # Momentum Acceleration (rate of change)
    data['mom_accel_3'] = data['mom_3'] - data['mom_3'].shift(3)
    data['mom_accel_10'] = data['mom_10'] - data['mom_10'].shift(5)
    
    # Volume-Price Fractal Sync
    # Volume Fractal Analysis
    def volume_fractal(volume_series, window=20):
        """Simplified volume fractal analysis"""
        vol_std = volume_series.rolling(window).std()
        vol_range = volume_series.rolling(window).max() - volume_series.rolling(window).min()
        return vol_std / (vol_range + 1e-8)
    
    data['vol_fractal_20'] = volume_fractal(data['volume'], 20)
    
    # Price-Volume Synchronization
    price_vol_corr = data['close'].pct_change().rolling(10).corr(data['volume'].pct_change())
    data['price_vol_sync'] = price_vol_corr.rolling(5).mean()
    
    # Momentum-Volume Divergence
    # Convergence Quality
    mom_vol_corr_3 = data['mom_3'].rolling(10).corr(data['vol_mom_3'])
    mom_vol_corr_10 = data['mom_10'].rolling(20).corr(data['vol_mom_10'])
    data['convergence_quality'] = (mom_vol_corr_3 + mom_vol_corr_10) / 2
    
    # Divergence Detection
    mom_trend = data['mom_10'].rolling(5).mean()
    vol_trend = data['vol_mom_10'].rolling(5).mean()
    data['momentum_divergence'] = np.sign(mom_trend) * np.sign(vol_trend) * (abs(mom_trend - vol_trend))
    
    # Regime-Dependent Signals
    # Fractal Regime Classification
    volatility_regime = np.where(data['vol_ratio_5_20'] > 1.2, 2, 
                                np.where(data['vol_ratio_5_20'] < 0.8, 0, 1))
    trend_regime = np.where(data['hurst_20'] > 0.6, 2, 
                           np.where(data['hurst_20'] < 0.4, 0, 1))
    data['fractal_regime'] = volatility_regime + trend_regime
    
    # Regime-Specific Filters
    regime_weights = {
        0: 0.3,  # Low volatility, anti-persistent
        1: 0.6,  # Normal regime
        2: 0.8,  # High volatility, trending
        3: 0.4,  # Mixed signals
        4: 1.0   # High volatility, trending (strongest)
    }
    
    # Signal Validation & Generation
    # Fractal Consistency Check
    multi_scale_consistency = (
        np.sign(data['mom_3']) * np.sign(data['mom_10']) * np.sign(data['mom_20'])
    )
    data['fractal_consistency'] = multi_scale_consistency.rolling(5).mean()
    
    # Final Signal Construction
    # Combine momentum components
    momentum_composite = (
        0.4 * data['mom_10'] + 
        0.3 * data['mom_20'] + 
        0.3 * data['asym_mom_ratio']
    )
    
    # Combine volume components
    volume_composite = (
        0.5 * data['vol_mom_10'] + 
        0.3 * data['vol_fractal_20'] + 
        0.2 * data['price_vol_sync']
    )
    
    # Combine with divergence and convergence
    divergence_component = (
        0.6 * data['convergence_quality'] + 
        0.4 * data['momentum_divergence']
    )
    
    # Final factor with regime adjustment
    base_factor = (
        0.5 * momentum_composite + 
        0.3 * volume_composite + 
        0.2 * divergence_component
    )
    
    # Apply regime-specific weights and consistency filter
    regime_weight = data['fractal_regime'].map(regime_weights)
    final_factor = base_factor * regime_weight * data['fractal_consistency']
    
    # Normalize and clean
    factor_series = final_factor.rolling(20).apply(
        lambda x: (x[-1] - np.nanmean(x)) / (np.nanstd(x) + 1e-8), 
        raw=True
    )
    
    return factor_series
