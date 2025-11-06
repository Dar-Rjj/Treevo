import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence factor combining fractal analysis with regime detection
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price Fractal Dimension Calculation
    def fractal_dimension(series, window=20):
        """Calculate fractal dimension using Hurst exponent approximation"""
        lags = range(2, min(10, window//2))
        tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    # Rolling price fractal dimension
    data['price_fractal'] = data['close'].rolling(window=20).apply(
        lambda x: fractal_dimension(x) if len(x) == 20 else np.nan, raw=False
    )
    
    # 2. Volume Fractal Dimension
    data['volume_fractal'] = data['volume'].rolling(window=20).apply(
        lambda x: fractal_dimension(x) if len(x) == 20 else np.nan, raw=False
    )
    
    # 3. Regime Detection using volatility and volume
    # Volatility regime (high/low volatility periods)
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    data['volume_ma_20d'] = data['volume'].rolling(window=20).mean()
    
    # Define regimes
    vol_threshold = data['volatility_20d'].rolling(window=60).quantile(0.7)
    vol_threshold = vol_threshold.fillna(method='ffill')
    
    volume_threshold = data['volume_ma_20d'].rolling(window=60).quantile(0.7)
    volume_threshold = volume_threshold.fillna(method='ffill')
    
    # High volatility regime
    data['high_vol_regime'] = (data['volatility_20d'] > vol_threshold).astype(int)
    # High volume regime
    data['high_volume_regime'] = (data['volume'] > volume_threshold).astype(int)
    
    # Combined regime indicator
    data['regime'] = data['high_vol_regime'] * 2 + data['high_volume_regime']
    
    # 4. Adaptive Momentum based on regime
    def adaptive_momentum(close, regime, short_period=5, long_period=20):
        momentum = np.zeros(len(close))
        for i in range(long_period, len(close)):
            if regime.iloc[i] == 0:  # Low vol, low volume - use longer momentum
                momentum[i] = (close.iloc[i] / close.iloc[i-long_period] - 1)
            elif regime.iloc[i] == 3:  # High vol, high volume - use shorter momentum
                momentum[i] = (close.iloc[i] / close.iloc[i-short_period] - 1)
            else:  # Mixed regimes - use medium momentum
                momentum[i] = (close.iloc[i] / close.iloc[i-10] - 1)
        return momentum
    
    data['adaptive_momentum'] = adaptive_momentum(data['close'], data['regime'])
    
    # 5. Microstructure noise extraction
    # Calculate intraday noise as the difference between close and VWAP-like measure
    data['intraday_vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['microstructure_noise'] = data['close'] - data['intraday_vwap']
    
    # Noise autocorrelation (1-day lag)
    data['noise_autocorr'] = data['microstructure_noise'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 10 else np.nan, raw=False
    )
    
    # 6. Volume-noise interaction
    data['volume_noise_corr'] = data['volume'].rolling(window=10).corr(data['microstructure_noise'])
    
    # 7. Temporal asymmetry - early vs late session momentum
    # Using OHLC to approximate session dynamics
    data['overnight_return'] = (data['open'] / data['close'].shift(1) - 1)
    data['intraday_return'] = (data['close'] / data['open'] - 1)
    
    # Early session momentum (first hour approximation using open-high range)
    data['early_strength'] = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['late_strength'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # 8. Final factor calculation - Price-Volume Fractal Divergence
    # Normalize components
    components = ['price_fractal', 'volume_fractal', 'adaptive_momentum', 
                 'microstructure_noise', 'noise_autocorr', 'volume_noise_corr',
                 'early_strength', 'late_strength']
    
    for col in components:
        if col in data.columns:
            data[f'{col}_norm'] = (data[col] - data[col].rolling(window=60).mean()) / data[col].rolling(window=60).std()
    
    # Combine components with weights
    factor = (
        0.25 * data['price_fractal_norm'].fillna(0) +
        0.20 * data['volume_fractal_norm'].fillna(0) +
        0.15 * data['adaptive_momentum'].fillna(0) +
        0.10 * data['microstructure_noise_norm'].fillna(0) +
        0.10 * data['noise_autocorr_norm'].fillna(0) +
        0.10 * data['volume_noise_corr_norm'].fillna(0) +
        0.05 * data['early_strength_norm'].fillna(0) +
        0.05 * data['late_strength_norm'].fillna(0)
    )
    
    # Apply regime-based smoothing
    regime_smooth = data['regime'].rolling(window=5).mean().fillna(0)
    smooth_factor = factor.rolling(window=3).mean() * (1 + 0.1 * regime_smooth)
    
    return smooth_factor
