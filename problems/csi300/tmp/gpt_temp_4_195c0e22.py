import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining price-volume fractal relationships and regime transition detection.
    The factor captures the multi-scale coupling between price movements and volume patterns,
    while adjusting for current market regime conditions.
    """
    # Calculate daily returns and price ranges
    returns = df['close'].pct_change()
    price_range = (df['high'] - df['low']) / df['close']
    
    # Multi-scale volume-price scaling (3 different time windows)
    windows = [5, 10, 20]
    volume_price_coupling = pd.Series(0.0, index=df.index)
    
    for window in windows:
        # Rolling correlation between absolute returns and volume (volume-price coupling)
        vol_price_corr = df['volume'].rolling(window).corr(abs(returns))
        # Rolling ratio of volume to price movement intensity
        vol_intensity_ratio = (df['volume'].rolling(window).mean() / 
                              (abs(returns).rolling(window).std() + 1e-8))
        
        volume_price_coupling += (vol_price_corr * vol_intensity_ratio) / len(windows)
    
    # Fractal dimension estimation of price-volume relationship
    # Using Hurst exponent-like calculation on volume-adjusted price movements
    def hurst_like_metric(series, max_lag=10):
        if len(series) < max_lag * 2:
            return np.nan
        lags = range(2, max_lag + 1)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Volume-adjusted price series for fractal analysis
    volume_weighted_returns = returns * np.log1p(df['volume'])
    fractal_dimension = volume_weighted_returns.rolling(30).apply(
        hurst_like_metric, raw=False, kwargs={'max_lag': 10}
    )
    
    # Regime transition detection - volatility and trend regimes
    volatility_regime = returns.rolling(20).std()
    trend_regime = df['close'].rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x), raw=False
    )
    
    # Normalize regimes to 0-1 scale
    vol_regime_norm = (volatility_regime - volatility_regime.rolling(50).min()) / \
                     (volatility_regime.rolling(50).max() - volatility_regime.rolling(50).min() + 1e-8)
    trend_regime_norm = (trend_regime - trend_regime.rolling(50).min()) / \
                       (trend_regime.rolling(50).max() - trend_regime.rolling(50).min() + 1e-8)
    
    # Combine fractal relationships with regime adjustments
    # Higher fractal dimension + strong volume-price coupling in low volatility regimes is favorable
    regime_adjusted_factor = (volume_price_coupling * fractal_dimension * 
                            (1 - vol_regime_norm) * np.sign(trend_regime_norm))
    
    # Final factor with smoothing and normalization
    factor = regime_adjusted_factor.rolling(5).mean()
    factor = (factor - factor.rolling(50).mean()) / (factor.rolling(50).std() + 1e-8)
    
    return factor
