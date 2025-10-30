import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Scaled Momentum with Volume Regime Confirmation
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate returns for different timeframes
    returns_3d = close.pct_change(periods=3)
    returns_10d = close.pct_change(periods=10)
    returns_20d = close.pct_change(periods=20)
    
    # Volatility calculations
    vol_5d = returns_3d.rolling(window=5).std()
    vol_20d = returns_10d.rolling(window=20).std()
    vol_60d = returns_20d.rolling(window=60).std()
    
    # Volatility-scaled momentum components
    mom_short = returns_3d / vol_5d.replace(0, np.nan)
    mom_medium = returns_10d / vol_20d.replace(0, np.nan)
    mom_long = returns_20d / vol_60d.replace(0, np.nan)
    
    # Volume trend analysis
    def calc_volume_slope(vol_series, window):
        slopes = pd.Series(index=vol_series.index, dtype=float)
        for i in range(window-1, len(vol_series)):
            if i >= window-1:
                y = vol_series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    vol_slope_5d = calc_volume_slope(volume, 5)
    vol_slope_20d = calc_volume_slope(volume, 20)
    
    # Normalize volume slopes by average volume
    vol_avg_5d = volume.rolling(window=5).mean()
    vol_avg_20d = volume.rolling(window=20).mean()
    
    vol_trend_short = vol_slope_5d / vol_avg_5d.replace(0, np.nan)
    vol_trend_medium = vol_slope_20d / vol_avg_20d.replace(0, np.nan)
    
    # Volume-momentum alignment scores
    vol_threshold = 0.01
    vol_confirm_short = np.where(
        abs(vol_trend_short) > vol_threshold,
        np.sign(mom_short) * np.sign(vol_trend_short),
        0
    )
    vol_confirm_medium = np.where(
        abs(vol_trend_medium) > vol_threshold,
        np.sign(mom_medium) * np.sign(vol_trend_medium),
        0
    )
    
    # Volatility regime detection
    vol_ratio = vol_5d / vol_20d.replace(0, np.nan)
    vol_regime = pd.Series('normal', index=close.index)
    vol_regime = np.where(vol_ratio > 1.5, 'high', vol_regime)
    vol_regime = np.where(vol_ratio < 0.67, 'low', vol_regime)
    
    # Timeframe weighting by regime
    weights = pd.DataFrame(index=close.index, columns=['short', 'medium', 'long'])
    weights['short'] = np.where(vol_regime == 'high', 0.6, 
                               np.where(vol_regime == 'low', 0.2, 0.33))
    weights['medium'] = np.where(vol_regime == 'high', 0.2, 
                                np.where(vol_regime == 'low', 0.2, 0.33))
    weights['long'] = np.where(vol_regime == 'high', 0.2, 
                              np.where(vol_regime == 'low', 0.6, 0.33))
    
    # Volume confirmation multiplier
    vol_multiplier = pd.Series(1.0, index=close.index)
    vol_multiplier = np.where((vol_confirm_short > 0) | (vol_confirm_medium > 0), 1.2, vol_multiplier)
    vol_multiplier = np.where((vol_confirm_short < 0) | (vol_confirm_medium < 0), 0.8, vol_multiplier)
    
    # Weighted sum of volatility-scaled momenta
    weighted_momentum = (
        weights['short'] * mom_short + 
        weights['medium'] * mom_medium + 
        weights['long'] * mom_long
    )
    
    # Apply volume confirmation multiplier
    momentum_signal = weighted_momentum * vol_multiplier
    
    # Mean reversion enhancement
    recent_range = (close.rolling(window=10).max() - close.rolling(window=10).min())
    price_position = (close - close.rolling(window=10).min()) / recent_range.replace(0, np.nan)
    
    # Detect extreme conditions
    overbought = price_position > 0.8
    oversold = price_position < 0.2
    
    # Apply mean reversion factor
    mean_rev_factor = pd.Series(1.0, index=close.index)
    mean_rev_factor = np.where(overbought, -0.3, mean_rev_factor)
    mean_rev_factor = np.where(oversold, 0.3, mean_rev_factor)
    
    # Final composite factor
    alpha_factor = momentum_signal * (1 + mean_rev_factor)
    
    return alpha_factor
