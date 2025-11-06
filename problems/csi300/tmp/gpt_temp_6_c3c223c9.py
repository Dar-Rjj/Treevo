import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Momentum-Volume Convergence Quality factor
    Multi-scale momentum and volume fractal analysis with regime-adaptive quality weighting
    """
    
    # Price Momentum Fractal Components
    df['mom_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['mom_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['mom_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volume Momentum Fractal Components
    df['vol_mom_3d'] = df['volume'] / df['volume'].shift(3) - 1
    df['vol_mom_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['vol_mom_20d'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Multi-Scale Volatility Fractal Analysis
    df['volatility_5d'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_20d'] = df['close'].pct_change().rolling(window=20).std()
    df['volatility_60d'] = df['close'].pct_change().rolling(window=60).std()
    
    df['vol_ratio_5_20'] = df['volatility_5d'] / df['volatility_20d']
    df['vol_ratio_20_60'] = df['volatility_20d'] / df['volatility_60d']
    
    # Price Fractal Dimension Estimation (Hurst Exponent approximation)
    def hurst_approximation(series, window=20):
        lags = range(2, min(10, window-1))
        tau = []
        for lag in lags:
            ts = series.rolling(window=lag).std()
            tau.append(np.log(ts.iloc[-1]) if not ts.empty and not np.isnan(ts.iloc[-1]) else 0)
        if len(tau) > 1:
            poly = np.polyfit(np.log(lags[:len(tau)]), tau, 1)
            return poly[0]
        return 0.5
    
    hurst_values = []
    for i in range(len(df)):
        if i >= 20:
            window_data = df['close'].iloc[i-19:i+1]
            hurst_values.append(hurst_approximation(window_data))
        else:
            hurst_values.append(0.5)
    df['hurst'] = hurst_values
    
    # Nonlinear Momentum-Volume Convergence Assessment
    # Asymmetric Momentum Response Analysis
    df['returns_5d'] = df['close'].pct_change(5)
    df['pos_mom_5d'] = df['returns_5d'].apply(lambda x: max(x, 0))
    df['neg_mom_5d'] = df['returns_5d'].apply(lambda x: min(x, 0))
    df['mom_asymmetry'] = (df['pos_mom_5d'] - df['neg_mom_5d']) / (df['pos_mom_5d'] - df['neg_mom_5d']).abs().replace(0, 1)
    
    # Volume-Price Fractal Synchronization
    # Calculate volume fractal dimension (simplified)
    vol_hurst_values = []
    for i in range(len(df)):
        if i >= 20:
            window_data = df['volume'].iloc[i-19:i+1]
            vol_hurst_values.append(hurst_approximation(window_data))
        else:
            vol_hurst_values.append(0.5)
    df['vol_hurst'] = vol_hurst_values
    
    df['fractal_sync'] = 1 - np.abs(df['hurst'] - df['vol_hurst'])
    
    # Breakout Efficiency with Fractal Validation
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['price_efficiency'] = (df['close'] - df['open']) / df['true_range'].replace(0, 1)
    
    df['vol_surge_20d'] = df['volume'] / df['volume'].rolling(window=20).mean()
    df['vol_mom_5d'] = df['volume'] / df['volume'].shift(5) - 1
    
    # Fractal Regime-Adaptive Quality Weighting
    high_fractal_regime = df['hurst'] > 0.6
    low_fractal_regime = df['hurst'] < 0.4
    
    # Momentum convergence strength
    df['mom_convergence'] = (
        np.sign(df['mom_3d']) * np.sign(df['mom_10d']) * np.sign(df['mom_20d']) *
        (np.abs(df['mom_3d']) + np.abs(df['mom_10d']) + np.abs(df['mom_20d'])) / 3
    )
    
    # Volume convergence strength
    df['vol_convergence'] = (
        np.sign(df['vol_mom_3d']) * np.sign(df['vol_mom_10d']) * np.sign(df['vol_mom_20d']) *
        (np.abs(df['vol_mom_3d']) + np.abs(df['vol_mom_10d']) + np.abs(df['vol_mom_20d'])) / 3
    )
    
    # Overall convergence strength
    df['convergence_strength'] = df['mom_convergence'] * df['vol_convergence']
    
    # Fractal-Enhanced Signal Generation
    # Base signal from momentum convergence
    base_signal = df['convergence_strength'] * df['fractal_sync']
    
    # Apply breakout efficiency multiplier
    breakout_multiplier = 1 + df['price_efficiency'] * df['vol_surge_20d']
    
    # Fractal regime adaptation
    regime_weight = np.where(
        high_fractal_regime,
        df['convergence_strength'] * df['fractal_sync'],  # Emphasize divergence in high fractal regime
        np.where(
            low_fractal_regime,
            df['convergence_strength'] * (1 + df['price_efficiency']),  # Emphasize persistence in low fractal regime
            df['convergence_strength']  # Neutral regime
        )
    )
    
    # Final factor calculation
    factor = base_signal * breakout_multiplier * regime_weight
    
    # Clean up intermediate columns
    cols_to_drop = ['mom_3d', 'mom_10d', 'mom_20d', 'vol_mom_3d', 'vol_mom_10d', 'vol_mom_20d',
                   'volatility_5d', 'volatility_20d', 'volatility_60d', 'vol_ratio_5_20', 'vol_ratio_20_60',
                   'hurst', 'returns_5d', 'pos_mom_5d', 'neg_mom_5d', 'mom_asymmetry', 'vol_hurst',
                   'fractal_sync', 'true_range', 'price_efficiency', 'vol_surge_20d', 'vol_mom_5d',
                   'mom_convergence', 'vol_convergence', 'convergence_strength']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return factor
