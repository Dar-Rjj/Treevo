import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Dynamics factor combining multi-scale volume-price divergence,
    volume profile asymmetry, and fractal dimension characteristics.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Multi-scale Volume-Price Divergence components
    # Short-term: 5-day volume surge without price movement
    volume_ma_5 = df['volume'].rolling(window=5, min_periods=3).mean()
    price_range_5 = (df['high'].rolling(window=5, min_periods=3).max() - 
                    df['low'].rolling(window=5, min_periods=3).min()) / df['close'].rolling(window=5, min_periods=3).mean()
    
    short_term_divergence = (df['volume'] / volume_ma_5) / (1 + price_range_5)
    
    # Medium-term: 20-day price trend with declining volume
    price_trend_20 = df['close'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] - x[0]) / x[0] if len(x) == 20 else np.nan, raw=True
    )
    volume_trend_20 = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] - x[0]) / x[0] if len(x) == 20 else np.nan, raw=True
    )
    medium_term_divergence = price_trend_20 * (1 - volume_trend_20)
    
    # Long-term: 60-day volume accumulation during sideways action
    price_volatility_60 = df['close'].rolling(window=60, min_periods=30).std() / df['close'].rolling(window=60, min_periods=30).mean()
    volume_accumulation_60 = df['volume'].rolling(window=60, min_periods=30).sum() / df['volume'].rolling(window=60, min_periods=30).mean()
    
    long_term_divergence = volume_accumulation_60 / (1 + price_volatility_60)
    
    # Volume Profile Asymmetry
    # Upper vs lower volume concentration bias
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    upper_half_volume = df['volume'] * ((df['high'] - typical_price) / (df['high'] - df['low'])).clip(0, 1)
    lower_half_volume = df['volume'] * ((typical_price - df['low']) / (df['high'] - df['low'])).clip(0, 1)
    
    volume_asymmetry = (upper_half_volume.rolling(window=10, min_periods=5).mean() - 
                       lower_half_volume.rolling(window=10, min_periods=5).mean()) / df['volume'].rolling(window=10, min_periods=5).mean()
    
    # Price-Volume Fractal Dimension approximation
    # Using Hurst exponent-like calculation for price and volume
    def hurst_like_approximation(series, window=20):
        lags = [2, 5, 10]
        rs_values = []
        for lag in lags:
            if lag < window:
                diff = series.diff(lag).dropna()
                if len(diff) > 0:
                    std_diff = diff.std()
                    if std_diff > 0:
                        rs = series.rolling(window=lag).std().dropna() / std_diff
                        if len(rs) > 0:
                            rs_values.append(rs.mean())
        
        if len(rs_values) > 1:
            return np.log(np.mean(rs_values)) / np.log(len(rs_values))
        return 1.0
    
    # Calculate fractal dimensions for price and volume
    price_fractal = df['close'].rolling(window=20, min_periods=10).apply(
        lambda x: hurst_like_approximation(pd.Series(x)) if len(x) == 20 else 1.0, raw=False
    )
    
    volume_fractal = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: hurst_like_approximation(pd.Series(x)) if len(x) == 20 else 1.0, raw=False
    )
    
    fractal_divergence = price_fractal - volume_fractal
    
    # Combine all components with appropriate weights
    factor = (
        0.3 * short_term_divergence +
        0.25 * medium_term_divergence +
        0.2 * long_term_divergence +
        0.15 * volume_asymmetry +
        0.1 * fractal_divergence
    )
    
    # Normalize the factor using rolling z-score (20-day window)
    factor_mean = factor.rolling(window=20, min_periods=10).mean()
    factor_std = factor.rolling(window=20, min_periods=10).std()
    normalized_factor = (factor - factor_mean) / factor_std
    
    return normalized_factor
