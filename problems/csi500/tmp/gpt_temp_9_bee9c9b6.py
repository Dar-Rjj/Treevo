import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate True Range
    daily_range = df['high'] - df['low']
    gap_up = abs(df['high'] - df['close'].shift(1))
    gap_down = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([daily_range, gap_up, gap_down], axis=1).max(axis=1)
    
    # Volatility Regime Classification
    rolling_vol = true_range.rolling(window=5, min_periods=5).sum() / 5
    median_true_range = true_range.rolling(window=20, min_periods=20).median()
    
    volatility_regime = pd.Series(index=df.index, dtype=str)
    high_vol_threshold = 1.5 * median_true_range
    low_vol_threshold = 0.7 * median_true_range
    
    volatility_regime = np.where(rolling_vol > high_vol_threshold, 'high',
                        np.where(rolling_vol < low_vol_threshold, 'low', 'normal'))
    
    # Momentum Calculations
    # Short-Term Momentum (3-day)
    short_return = df['close'] / df['close'].shift(3) - 1
    short_vol_scaling = true_range.rolling(window=5, min_periods=5).sum() / 5 + 0.0001
    short_momentum = short_return / short_vol_scaling
    
    # Medium-Term Momentum (8-day)
    medium_return = df['close'] / df['close'].shift(8) - 1
    medium_returns = df['close'].rolling(window=8).apply(
        lambda x: (x / x.iloc[0]).std() if len(x) == 8 else np.nan, raw=False
    )
    medium_vol_scaling = medium_returns + 0.0001
    medium_momentum = medium_return / medium_vol_scaling
    
    # Long-Term Momentum (20-day)
    long_return = df['close'] / df['close'].shift(20) - 1
    long_returns = df['close'].rolling(window=20).apply(
        lambda x: (x / x.iloc[0]).std() if len(x) == 20 else np.nan, raw=False
    )
    long_vol_scaling = long_returns + 0.0001
    long_momentum = long_return / long_vol_scaling
    
    # Volume Calculations
    volume_ratio = df['volume'] / df['volume'].shift(3)
    volume_slope = (df['volume'] - df['volume'].shift(8)) / 8
    volume_acceleration = volume_ratio - (df['volume'].shift(1) / df['volume'].shift(4))
    
    # Volume Regime Classification
    median_volume = df['volume'].rolling(window=20, min_periods=20).median()
    high_vol_threshold_vol = 1.5 * median_volume
    low_vol_threshold_vol = 0.7 * median_volume
    
    volume_regime = np.where(df['volume'] > high_vol_threshold_vol, 'high',
                    np.where(df['volume'] < low_vol_threshold_vol, 'low', 'normal'))
    
    # Adaptive Factor Integration
    # Momentum Composite
    momentum_composite = (np.sign(short_momentum) * 
                         np.abs(short_momentum) ** 0.4 * 
                         np.abs(medium_momentum) ** 0.35 * 
                         np.abs(long_momentum) ** 0.25)
    
    # Volume Confirmation
    volume_mean_20d = df['volume'].rolling(window=20, min_periods=20).mean()
    volume_slope_normalized = np.minimum(np.abs(volume_slope) / volume_mean_20d, 1)
    volume_confirmation = volume_ratio * (1 + np.sign(volume_slope) * volume_slope_normalized)
    
    # Combined Signal
    combined_signal = momentum_composite * volume_confirmation
    
    # Regime-Adaptive Scaling
    # Volatility Regime Adjustment
    vol_adjustment = np.where(volatility_regime == 'high', 0.7,
                     np.where(volatility_regime == 'low', 1.3, 1.0))
    
    vol_adjusted_signal = combined_signal * vol_adjustment
    
    # Volume Regime Adjustment
    volume_adjustment = np.where(volume_regime == 'high', 1.2,
                        np.where(volume_regime == 'low', 0.8, 1.0))
    
    final_adjusted_signal = vol_adjusted_signal * volume_adjustment
    
    # Final Alpha Factor with capping
    result = np.sign(final_adjusted_signal) * np.minimum(np.abs(final_adjusted_signal), 2)
    
    return result
