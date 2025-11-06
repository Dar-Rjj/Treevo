import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence with Momentum Decay and Regime Persistence
    Multi-scale alpha factor combining fractal efficiency, momentum decay, gap dynamics, and correlation alignment
    """
    data = df.copy()
    
    # Multi-Scale Fractal Efficiency with Volume Asymmetry
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using rescaled range analysis"""
        if len(series) < window:
            return np.nan
        
        # Calculate log returns
        log_returns = np.log(series / series.shift(1)).dropna()
        if len(log_returns) < window:
            return np.nan
            
        # Mean-adjusted series
        mean_adjusted = log_returns - log_returns.mean()
        
        # Cumulative deviation
        cumulative_dev = mean_adjusted.cumsum()
        
        # Range
        R = cumulative_dev.max() - cumulative_dev.min()
        
        # Standard deviation
        S = log_returns.std()
        
        if S == 0:
            return 0.5
            
        return np.log(R / S) / np.log(window)
    
    # Calculate fractal dimensions across horizons
    data['fractal_5d'] = data['close'].rolling(window=5).apply(
        lambda x: hurst_exponent(x, 5), raw=False)
    data['fractal_15d'] = data['close'].rolling(window=15).apply(
        lambda x: hurst_exponent(x, 15), raw=False)
    data['fractal_30d'] = data['close'].rolling(window=30).apply(
        lambda x: hurst_exponent(x, 30), raw=False)
    
    # Volume asymmetry patterns
    data['returns'] = data['close'].pct_change()
    data['up_day'] = data['returns'] > 0
    data['down_day'] = data['returns'] < 0
    
    # Up-day volume concentration
    data['up_volume'] = np.where(data['up_day'], data['volume'], 0)
    data['total_volume_10d'] = data['volume'].rolling(window=10).sum()
    data['up_volume_10d'] = data['up_volume'].rolling(window=10).sum()
    data['volume_concentration'] = data['up_volume_10d'] / data['total_volume_10d']
    
    # Down-day volume intensity
    data['down_volume'] = np.where(data['down_day'], data['volume'], 0)
    data['avg_up_volume'] = data['up_volume'].rolling(window=10).mean()
    data['avg_down_volume'] = data['down_volume'].rolling(window=10).mean()
    data['volume_intensity'] = data['avg_down_volume'] / data['avg_up_volume']
    
    # Volume fractal dimension
    data['volume_fractal'] = data['volume'].rolling(window=10).apply(
        lambda x: hurst_exponent(x, 10), raw=False)
    
    # Fractal efficiency ratio
    data['fractal_efficiency'] = (
        data['fractal_15d'] * data['volume_concentration'] + 
        (1 - data['fractal_15d']) * (1 - data['volume_intensity'])
    )
    
    # Momentum Decay Acceleration with Regime Transition Detection
    # Short-term decay
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_6d'] = data['close'].shift(3) / data['close'].shift(6) - 1
    data['decay_short'] = data['momentum_3d'] - data['momentum_6d']
    
    # Medium-term decay
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_20d'] = data['close'].shift(10) / data['close'].shift(20) - 1
    data['decay_medium'] = data['momentum_10d'] - data['momentum_20d']
    
    # Long-term decay
    data['momentum_20d_curr'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_40d'] = data['close'].shift(20) / data['close'].shift(40) - 1
    data['decay_long'] = data['momentum_20d_curr'] - data['momentum_40d']
    
    # Regime transition signals
    data['daily_range'] = data['high'] - data['low']
    data['range_median_20d'] = data['daily_range'].rolling(window=20).median()
    data['volatility_regime'] = data['daily_range'] / data['range_median_20d']
    
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_regime'] = data['volume'] / data['volume_ma_10d']
    
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['price_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    
    # Decay-adjusted momentum
    data['decay_composite'] = (
        data['decay_short'] * 0.4 + 
        data['decay_medium'] * 0.35 + 
        data['decay_long'] * 0.25
    )
    
    # Gap-Cluster Persistence with Intraday Range Compression
    data['gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_magnitude'] = data['gap'].abs()
    
    # Gap clustering (3-day similarity)
    data['gap_cluster'] = (
        data['gap_magnitude'].rolling(window=3).std() / 
        data['gap_magnitude'].rolling(window=3).mean()
    )
    
    # Gap direction persistence
    data['gap_sign'] = np.sign(data['gap'])
    data['gap_persistence'] = data['gap_sign'].rolling(window=3).apply(
        lambda x: len(set(x)) == 1, raw=False
    ).astype(float)
    
    # Range compression
    data['range_ma_5d'] = data['daily_range'].rolling(window=5).mean()
    data['range_compression'] = data['daily_range'] / data['range_ma_5d'].shift(1)
    
    # Opening range efficiency
    data['open_low_range'] = data['open'] - data['low']
    data['high_open_range'] = data['high'] - data['open']
    data['opening_efficiency'] = data['open_low_range'] / (data['high_open_range'] + 1e-8)
    
    # Gap-cluster range signals
    data['gap_cluster_signal'] = (
        data['gap_persistence'] * (1 - data['range_compression']) +
        (1 - data['gap_persistence']) * data['range_compression']
    )
    
    # Price-Volume Fractal Correlation with Momentum Alignment
    # Short-term correlation (5-day)
    data['price_vol_corr_5d'] = data['close'].rolling(window=5).corr(data['volume'])
    
    # Medium-term correlation (15-day)
    data['price_vol_corr_15d'] = data['close'].rolling(window=15).corr(data['volume'])
    
    # Long-term correlation (30-day)
    data['price_vol_corr_30d'] = data['close'].rolling(window=30).corr(data['volume'])
    
    # Momentum-fractal alignment
    data['momentum_fractal_alignment'] = (
        data['momentum_10d'] * data['fractal_15d'] * 
        np.sign(data['price_vol_corr_15d'])
    )
    
    # Fractal-momentum composite
    data['fractal_momentum_composite'] = (
        data['momentum_fractal_alignment'] * 
        (1 - np.abs(data['decay_medium'])) * 
        np.abs(data['price_vol_corr_15d'])
    )
    
    # Synthesize Multi-Dimensional Fractal Alpha
    # Combine fractal efficiency with momentum decay
    fractal_momentum = (
        data['fractal_efficiency'] * 
        (1 - data['decay_composite'].abs()) *
        data['volatility_regime']
    )
    
    # Integrate gap-cluster dynamics
    gap_range_component = (
        data['gap_cluster_signal'] * 
        data['volume_concentration'] *
        data['opening_efficiency']
    )
    
    # Apply fractal correlation alignment
    correlation_component = (
        data['fractal_momentum_composite'] * 
        np.sign(data['price_vol_corr_15d']) *
        (1 + data['fractal_30d'])
    )
    
    # Final alpha synthesis
    alpha = (
        fractal_momentum * 0.4 +
        gap_range_component * 0.35 +
        correlation_component * 0.25
    )
    
    return alpha
