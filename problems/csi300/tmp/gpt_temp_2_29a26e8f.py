import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Adjusted Momentum Fractality factor
    Combines momentum across different time scales with volatility adjustment and fractal analysis
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate multi-scale momentum (using 15, 30, 60 minute periods)
    # Assuming data is at 1-minute frequency, adjust shifts accordingly
    data['momentum_15'] = data['close'] - data['close'].shift(15)
    data['momentum_30'] = data['close'] - data['close'].shift(30)
    data['momentum_60'] = data['close'] - data['close'].shift(60)
    
    # Calculate corresponding volatility measures (range-based)
    data['volatility_15'] = (data['high'].rolling(window=15).max() - 
                           data['low'].rolling(window=15).min()) / data['close'].rolling(window=15).mean()
    data['volatility_30'] = (data['high'].rolling(window=30).max() - 
                           data['low'].rolling(window=30).min()) / data['close'].rolling(window=30).mean()
    data['volatility_60'] = (data['high'].rolling(window=60).max() - 
                           data['low'].rolling(window=60).min()) / data['close'].rolling(window=60).mean()
    
    # Volatility-adjusted momentum
    data['adj_momentum_15'] = data['momentum_15'] / (data['volatility_15'] + 1e-8)
    data['adj_momentum_30'] = data['momentum_30'] / (data['volatility_30'] + 1e-8)
    data['adj_momentum_60'] = data['momentum_60'] / (data['volatility_60'] + 1e-8)
    
    # Calculate fractal dimension using Hurst exponent approximation
    def hurst_exponent(series, max_lag=50):
        if len(series) < max_lag:
            return 0.5
        lags = range(2, min(max_lag, len(series)//2))
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Calculate fractal efficiency (path efficiency)
    def fractal_efficiency(series, window=60):
        if len(series) < window:
            return 0
        actual_path = np.sum(np.abs(np.diff(series[-window:])))
        straight_distance = np.abs(series[-1] - series[-window])
        return straight_distance / (actual_path + 1e-8) if actual_path > 0 else 0
    
    # Rolling calculations for fractal measures
    data['hurst_adj_momentum'] = data['adj_momentum_15'].rolling(window=100).apply(
        lambda x: hurst_exponent(x.dropna()), raw=False
    )
    
    data['fractal_efficiency'] = data['adj_momentum_15'].rolling(window=60).apply(
        lambda x: fractal_efficiency(x.dropna()), raw=False
    )
    
    # Regime identification
    data['momentum_strength'] = (data['adj_momentum_15'].abs() + 
                               data['adj_momentum_30'].abs() + 
                               data['adj_momentum_60'].abs()) / 3
    
    data['fractality_level'] = 1 - data['fractal_efficiency']
    
    # Regime classification
    conditions = [
        (data['momentum_strength'] > data['momentum_strength'].rolling(100).mean()) & 
        (data['fractality_level'] < data['fractality_level'].rolling(100).mean()),
        (data['momentum_strength'] < data['momentum_strength'].rolling(100).mean()) & 
        (data['fractality_level'] > data['fractality_level'].rolling(100).mean()),
        (data['momentum_strength'].abs() > data['momentum_strength'].rolling(100).std()) & 
        (data['fractality_level'] < 0.3)
    ]
    
    choices = [2, -2, 3]  # Strong trend, chaotic, very strong trend
    data['regime_signal'] = np.select(conditions, choices, default=1)
    
    # Scale-weighted signals
    volatility_stability_15 = 1 / (1 + data['volatility_15'].rolling(30).std())
    volatility_stability_30 = 1 / (1 + data['volatility_30'].rolling(30).std())
    volatility_stability_60 = 1 / (1 + data['volatility_60'].rolling(30).std())
    
    # Weighted momentum signal
    weighted_momentum = (
        data['adj_momentum_15'] * volatility_stability_15 +
        data['adj_momentum_30'] * volatility_stability_30 +
        data['adj_momentum_60'] * volatility_stability_60
    ) / (volatility_stability_15 + volatility_stability_30 + volatility_stability_60 + 1e-8)
    
    # Final factor combining momentum, fractality, and regime information
    data['factor'] = (
        weighted_momentum * 
        data['regime_signal'] * 
        data['fractal_efficiency'] *
        (1 - data['fractality_level'])
    )
    
    # Normalize the factor
    data['factor'] = (data['factor'] - data['factor'].rolling(252).mean()) / data['factor'].rolling(252).std()
    
    return data['factor']
