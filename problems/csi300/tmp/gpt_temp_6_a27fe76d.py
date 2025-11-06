import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor using multi-dimensional volume-momentum alignment and regime detection
    """
    # Volume-Momentum Alignment Spectrum
    # Short-term alignment (3-day)
    price_return_3d = df['close'].pct_change(3)
    volume_return_3d = df['volume'].pct_change(3)
    short_term_alignment = np.sign(price_return_3d) * np.sign(volume_return_3d) * np.minimum(np.abs(price_return_3d), np.abs(volume_return_3d))
    
    # Medium-term alignment (10-day)
    price_return_10d = df['close'].pct_change(10)
    volume_return_10d = df['volume'].pct_change(10)
    medium_term_alignment = np.sign(price_return_10d) * np.sign(volume_return_10d) * np.minimum(np.abs(price_return_10d), np.abs(volume_return_10d))
    
    # Alignment divergence signal
    alignment_divergence = short_term_alignment - medium_term_alignment
    
    # Volatility-Regime Adaptive Momentum
    # Volatility regime classification
    prev_close = df['close'].shift(1)
    true_range = np.maximum(df['high'] - df['low'], 
                           np.maximum(np.abs(df['high'] - prev_close), 
                                     np.abs(df['low'] - prev_close)))
    median_true_range = true_range.rolling(window=20, min_periods=10).median()
    regime_multiplier = true_range / median_true_range
    
    # Multi-timeframe momentum calculation
    short_term_momentum = df['close'].pct_change(5)
    medium_term_momentum = df['close'].pct_change(20)
    momentum_ratio = short_term_momentum / medium_term_momentum
    
    # Regime-adaptive signal
    regime_adaptive_momentum = momentum_ratio * regime_multiplier
    
    # Multi-Timeframe Efficiency Convergence
    # Intraday price efficiency
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Overnight gap efficiency
    prev_close = df['close'].shift(1)
    gap_absorption = np.abs(df['open'] - prev_close) / (df['high'] - df['low'])
    overnight_efficiency = 1 - gap_absorption  # Inverse relationship
    
    # Efficiency convergence signal
    efficiency_divergence = intraday_efficiency - overnight_efficiency
    
    # Volume-Weighted Momentum Acceleration
    # Price acceleration component
    momentum_5d = df['close'].pct_change(5)
    momentum_10d = df['close'].pct_change(10)
    price_acceleration = momentum_5d - momentum_10d
    
    # Volume confirmation filter
    median_volume_20d = df['volume'].rolling(window=20, min_periods=10).median()
    volume_spike_ratio = df['volume'] / median_volume_20d
    
    # Volume-weighted acceleration
    volume_weighted_acceleration = price_acceleration * volume_spike_ratio
    
    # Auction Theory Pressure Alignment
    # Opening session pressure
    opening_drive = (df['high'] - df['open']) - (df['open'] - df['low'])
    
    # Closing session pressure
    hl_midpoint = (df['high'] + df['low']) / 2
    closing_drive = (df['close'] - hl_midpoint) / (df['high'] - df['low'])
    
    # Session pressure alignment
    pressure_convergence = closing_drive - opening_drive
    
    # Volume-Price Divergence Spectrum
    # Short-term divergence (5-day)
    def linear_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    price_slope_5d = linear_slope(df['close'], 5)
    volume_slope_5d = linear_slope(df['volume'], 5)
    short_term_divergence = price_slope_5d - volume_slope_5d
    
    # Medium-term divergence (20-day)
    price_slope_20d = linear_slope(df['close'], 20)
    volume_slope_20d = linear_slope(df['volume'], 20)
    medium_term_divergence = price_slope_20d - volume_slope_20d
    
    # Multi-timeframe divergence signal
    divergence_ratio = short_term_divergence / medium_term_divergence
    
    # Combine all components with equal weights
    factor = (alignment_divergence.fillna(0) + 
              regime_adaptive_momentum.fillna(0) + 
              efficiency_divergence.fillna(0) + 
              volume_weighted_acceleration.fillna(0) + 
              pressure_convergence.fillna(0) + 
              divergence_ratio.fillna(0)) / 6
    
    return factor
