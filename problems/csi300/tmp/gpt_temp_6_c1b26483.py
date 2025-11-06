import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Price momentum and volume acceleration components
    price_momentum = close.pct_change(periods=5)
    volume_acceleration = volume.pct_change(periods=3)
    
    # Adaptive rolling correlation window based on volatility regime
    volatility_regime = close.rolling(window=20).std() / close.rolling(window=20).mean()
    dynamic_window = np.where(volatility_regime > volatility_regime.rolling(window=50).median(), 10, 20)
    
    # Regime-dependent correlation structure
    correlation_matrix = pd.Series(index=close.index, dtype=float)
    for i in range(max(dynamic_window), len(close)):
        window_size = int(dynamic_window[i])
        start_idx = i - window_size + 1
        if start_idx < 0:
            continue
        window_pm = price_momentum.iloc[start_idx:i+1]
        window_va = volume_acceleration.iloc[start_idx:i+1]
        valid_mask = (~window_pm.isna()) & (~window_va.isna())
        if valid_mask.sum() > 3:
            corr = window_pm[valid_mask].corr(window_va[valid_mask])
            correlation_matrix.iloc[i] = corr if not np.isnan(corr) else 0
    
    # Volatility state filtering
    regime_filter = np.where(volatility_regime > volatility_regime.rolling(window=50).quantile(0.7), 
                            correlation_matrix * 0.5, correlation_matrix * 1.5)
    
    # Momentum-volume convergence factor
    convergence_factor = regime_filter * price_momentum * volume_acceleration
    heuristics_matrix = convergence_factor.rolling(window=8).mean()
    
    return heuristics_matrix
