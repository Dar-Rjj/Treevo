import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Coherence & Regime Asymmetry Analysis Alpha Factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Price Fractal Analysis
    # Daily Price Range Efficiency
    data['price_range_efficiency'] = (data['high'] - data['low']) / (np.abs(data['open'] - data['close']) + 1e-8)
    
    # Calculate multi-scale fractal dimensions using Hurst-like approach
    def calculate_fractal_dimension(series, window):
        if len(series) < window:
            return np.nan
        log_returns = np.log(series / series.shift(1)).dropna()
        if len(log_returns) < window:
            return np.nan
        # Simplified fractal dimension estimation
        range_series = series.rolling(window).max() - series.rolling(window).min()
        std_series = series.rolling(window).std()
        fractal_dim = 2 - (np.log(range_series / std_series) / np.log(window))
        return fractal_dim
    
    # Price fractal dimensions
    for window in [5, 10, 20]:
        data[f'price_fractal_{window}'] = data['close'].rolling(window).apply(
            lambda x: calculate_fractal_dimension(x, window), raw=False
        )
    
    # Price Fractal Efficiency Scores
    data['fractal_efficiency_5'] = data['price_fractal_5'] / (data['price_fractal_20'] + 1e-8)
    data['fractal_efficiency_10'] = data['price_fractal_10'] / (data['price_fractal_20'] + 1e-8)
    
    # Volume Fractal Coherence
    # Multi-timeframe Volume Fractal Dimensions
    for window in [5, 10, 20]:
        data[f'volume_fractal_{window}'] = data['volume'].rolling(window).apply(
            lambda x: calculate_fractal_dimension(x, window), raw=False
        )
    
    # Price-Volume Fractal Coherence Comparison
    data['pv_fractal_coherence_5'] = data['price_fractal_5'] / (data['volume_fractal_5'] + 1e-8)
    data['pv_fractal_coherence_10'] = data['price_fractal_10'] / (data['volume_fractal_10'] + 1e-8)
    data['pv_fractal_coherence_20'] = data['price_fractal_20'] / (data['volume_fractal_20'] + 1e-8)
    
    # Asymmetric Return Analysis
    # Upside/Downside Volatility Ratio
    returns = data['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(20, min_periods=5).std()
    downside_vol = returns[returns < 0].rolling(20, min_periods=5).std()
    data['volatility_asymmetry'] = upside_vol / (downside_vol + 1e-8)
    
    # Return Path Directional Persistence
    data['return_persistence'] = returns.rolling(10).apply(
        lambda x: len(np.where(np.diff(np.sign(x)) != 0)[0]) / max(len(x) - 1, 1), raw=False
    )
    
    # Volume-Return Asymmetry Patterns
    volume_returns = data['volume'].pct_change()
    data['volume_return_corr'] = returns.rolling(10).corr(volume_returns)
    
    # Regime-Dependent Behavior
    # Volatility clustering
    volatility_regime = data['close'].pct_change().rolling(20).std()
    data['high_vol_regime'] = (volatility_regime > volatility_regime.rolling(60).quantile(0.7)).astype(int)
    
    # Volume intensity regime
    volume_zscore = (data['volume'] - data['volume'].rolling(60).mean()) / data['volume'].rolling(60).std()
    data['high_volume_regime'] = (volume_zscore > 1).astype(int)
    
    # Volatility-Clustered Fractal Analysis
    data['fractal_vol_adjusted'] = data['price_fractal_10'] * (1 + data['high_vol_regime'] * 0.5)
    
    # Price-Volume Momentum Divergence
    # Multi-scale Price Momentum
    for window in [5, 10, 20]:
        data[f'price_momentum_{window}'] = data['close'] / data['close'].shift(window) - 1
    
    # Volume Momentum Assessment
    for window in [5, 10, 20]:
        data[f'volume_momentum_{window}'] = data['volume'] / data['volume'].shift(window) - 1
    
    # Momentum Direction Mismatch Detection
    data['momentum_divergence_5'] = np.sign(data['price_momentum_5']) != np.sign(data['volume_momentum_5'])
    data['momentum_divergence_10'] = np.sign(data['price_momentum_10']) != np.sign(data['volume_momentum_10'])
    
    # Fractal Coherence Breakpoints
    # Local Fractal Dimension Change Detection
    data['fractal_change_5'] = data['price_fractal_5'].diff(3)
    data['fractal_change_10'] = data['price_fractal_10'].diff(5)
    
    # Regime Transition Signal
    data['regime_transition'] = (data['high_vol_regime'].diff() != 0) | (data['high_volume_regime'].diff() != 0)
    
    # Composite Alpha Generation
    # Base Fractal Coherence Score
    base_coherence = (
        data['pv_fractal_coherence_10'].rolling(5).mean() * 
        data['fractal_efficiency_10']
    )
    
    # Momentum Divergence Enhancement
    momentum_enhancement = (
        data['momentum_divergence_5'].astype(float) * 0.3 + 
        data['momentum_divergence_10'].astype(float) * 0.7
    )
    
    # Asymmetric Pattern Integration
    asymmetric_component = (
        data['volatility_asymmetry'].rolling(5).mean() * 
        (1 - data['return_persistence'])
    )
    
    # Regime Adjustment
    regime_adjustment = np.where(
        data['high_vol_regime'] == 1,
        1.2,  # Boost signal in high volatility
        np.where(data['high_volume_regime'] == 1, 0.8, 1.0)  # Reduce in high volume
    )
    
    # Confidence-Weighted Aggregation
    fractal_confidence = 1 / (1 + np.abs(data['fractal_change_10']))
    
    # Final composite alpha
    alpha = (
        base_coherence * 0.4 +
        momentum_enhancement * 0.25 +
        asymmetric_component * 0.2 +
        data['fractal_vol_adjusted'] * 0.15
    ) * regime_adjustment * fractal_confidence
    
    # Handle regime transitions
    alpha = np.where(data['regime_transition'], alpha * 0.7, alpha)
    
    return alpha
