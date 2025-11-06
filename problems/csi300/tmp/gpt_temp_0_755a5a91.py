import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Momentum Regime Factor combining multi-scale volatility, 
    price/volume fractal analysis, asymmetric momentum, and regime classification
    """
    # Multi-Scale Volatility Fractals
    df['vol_5d'] = df['close'].pct_change().rolling(window=5).std()
    df['vol_20d'] = df['close'].pct_change().rolling(window=20).std()
    df['vol_60d'] = df['close'].pct_change().rolling(window=60).std()
    
    df['vol_ratio_5_20'] = df['vol_5d'] / df['vol_20d']
    df['vol_ratio_20_60'] = df['vol_20d'] / df['vol_60d']
    
    # Price Fractal Dimension - Hurst Exponent approximation
    def hurst_exponent(series, window=20):
        lags = range(2, min(window, len(series)))
        tau = []
        for lag in lags:
            if len(series) >= lag:
                tau.append(np.std(np.diff(series, lag)))
            else:
                tau.append(np.nan)
        
        if len(tau) > 1:
            lags = np.array(lags[:len(tau)])
            tau = np.array(tau)
            mask = ~np.isnan(tau) & (tau > 0)
            if np.sum(mask) > 1:
                return np.polyfit(np.log(lags[mask]), np.log(tau[mask]), 1)[0]
        return np.nan
    
    df['hurst_price'] = df['close'].rolling(window=20).apply(
        lambda x: hurst_exponent(x) if len(x) == 20 else np.nan, raw=False
    )
    
    # Price Path Complexity (simplified)
    df['price_complexity'] = (df['high'] - df['low']).rolling(window=5).std() / df['close'].rolling(window=5).std()
    
    # Asymmetric Momentum
    df['returns_5d'] = df['close'].pct_change(5)
    df['upside_momentum'] = df['returns_5d'].where(df['returns_5d'] > 0, 0)
    df['downside_momentum'] = -df['returns_5d'].where(df['returns_5d'] < 0, 0)
    df['momentum_asymmetry'] = df['upside_momentum'] / (df['downside_momentum'] + 1e-8)
    
    # Momentum Acceleration
    df['momentum_3d'] = df['close'].pct_change(3)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_accel_3d'] = df['momentum_3d'] / (df['close'].pct_change(6) + 1e-8)
    df['momentum_accel_10d'] = df['momentum_10d'] / (df['close'].pct_change(20) + 1e-8)
    
    # Volume Fractal Analysis
    df['hurst_volume'] = df['volume'].rolling(window=20).apply(
        lambda x: hurst_exponent(x) if len(x) == 20 else np.nan, raw=False
    )
    
    # Volume Clustering Patterns
    df['volume_clustering'] = df['volume'].rolling(window=10).apply(
        lambda x: np.mean((x - x.mean())**3) / (x.std()**3 + 1e-8), raw=False
    )
    
    # Price-Volume Fractal Sync
    df['fractal_dim_diff'] = df['hurst_price'] - df['hurst_volume']
    df['fractal_sync'] = 1 / (1 + np.abs(df['fractal_dim_diff']))
    
    # Regime Classification
    vol_condition = (df['vol_ratio_5_20'] > 1.2) & (df['vol_ratio_20_60'] > 1.1)
    hurst_condition = df['hurst_price'] > 0.6
    complexity_condition = df['price_complexity'] > df['price_complexity'].rolling(20).mean()
    
    df['high_fractal_regime'] = (vol_condition & hurst_condition & complexity_condition).astype(int)
    df['low_fractal_regime'] = ((df['vol_ratio_5_20'] < 0.8) & (df['hurst_price'] < 0.4)).astype(int)
    df['transition_regime'] = (~df['high_fractal_regime'].astype(bool) & ~df['low_fractal_regime'].astype(bool)).astype(int)
    
    # Signal Integration
    # Regime-Weighted Momentum
    regime_weights = (
        df['high_fractal_regime'] * 0.3 + 
        df['low_fractal_regime'] * 0.6 + 
        df['transition_regime'] * 0.1
    )
    
    composite_momentum = (
        df['momentum_asymmetry'] * 0.4 +
        df['momentum_accel_3d'] * 0.3 +
        df['momentum_accel_10d'] * 0.3
    )
    
    df['regime_weighted_momentum'] = regime_weights * composite_momentum
    
    # Fractal Sync Multiplier
    fractal_multiplier = 1 + df['fractal_sync'] * 0.5
    
    # Final Alpha Score
    alpha_score = df['regime_weighted_momentum'] * fractal_multiplier
    
    # Clean up intermediate columns
    cols_to_drop = ['vol_5d', 'vol_20d', 'vol_60d', 'vol_ratio_5_20', 'vol_ratio_20_60',
                   'hurst_price', 'price_complexity', 'returns_5d', 'upside_momentum',
                   'downside_momentum', 'momentum_asymmetry', 'momentum_3d', 'momentum_10d',
                   'momentum_accel_3d', 'momentum_accel_10d', 'hurst_volume', 
                   'volume_clustering', 'fractal_dim_diff', 'fractal_sync',
                   'high_fractal_regime', 'low_fractal_regime', 'transition_regime',
                   'regime_weighted_momentum']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return alpha_score
