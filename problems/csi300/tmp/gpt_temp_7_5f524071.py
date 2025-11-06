import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Fractal Range Momentum Divergence factor
    Combines multi-timeframe volatility fractal analysis with range-momentum dynamics
    and volume-confirmed pressure signals for regime-adaptive alpha generation
    """
    
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Multi-Timeframe Volatility Fractal Analysis
    vol_10d = returns.rolling(window=10, min_periods=5).std()
    
    # Volatility fractal dimension across multiple periods
    def calc_fractal_dim(vol_series, window):
        """Calculate approximate fractal dimension using volatility scaling"""
        if len(vol_series) < window:
            return np.nan
        window_data = vol_series[-window:]
        if window_data.isna().any():
            return np.nan
        # Simple fractal dimension approximation using volatility scaling
        log_range = np.log(window_data.max() - window_data.min() + 1e-8)
        log_window = np.log(window)
        return 2 - (log_range / log_window) if log_window > 0 else 1.0
    
    # Compute fractal dimensions for different periods
    fractal_3d = vol_10d.rolling(window=3, min_periods=2).apply(
        lambda x: calc_fractal_dim(x, 3), raw=False
    )
    fractal_5d = vol_10d.rolling(window=5, min_periods=3).apply(
        lambda x: calc_fractal_dim(x, 5), raw=False
    )
    fractal_8d = vol_10d.rolling(window=8, min_periods=4).apply(
        lambda x: calc_fractal_dim(x, 8), raw=False
    )
    
    # Volatility regime transitions using fractal breaks
    fractal_break = (fractal_3d.diff(1) > fractal_3d.rolling(5).std()) | \
                   (fractal_5d.diff(1) > fractal_5d.rolling(5).std())
    
    # Range-Momentum Confluence Detection
    daily_range = (df['high'] - df['low']) / df['close']
    vol_adjusted_range = daily_range / (vol_10d + 1e-8)
    
    # Momentum persistence within range clusters
    momentum_5d = df['close'].pct_change(5)
    range_momentum_divergence = momentum_5d - vol_adjusted_range.rolling(5).mean()
    
    # Volume-Confirmed Pressure Dynamics
    high_close_pressure = (df['high'] - df['close']) / df['close']
    close_low_pressure = (df['close'] - df['low']) / df['close']
    directional_pressure = high_close_pressure - close_low_pressure
    
    # Volume-weighted pressure gradient
    volume_weighted_pressure = (directional_pressure * df['volume']).rolling(3).mean()
    pressure_volume_divergence = directional_pressure - volume_weighted_pressure / (df['volume'].rolling(3).mean() + 1e-8)
    
    # Adaptive Fractal-Regime Signals
    high_vol_regime = vol_10d > vol_10d.rolling(20).median()
    low_vol_regime = vol_10d < vol_10d.rolling(20).median()
    
    # Breakout momentum signal
    breakout_signal = (high_vol_regime & 
                      (fractal_3d > fractal_5d) & 
                      (daily_range > daily_range.rolling(10).mean()))
    
    # Accumulation signal
    accumulation_signal = (low_vol_regime & 
                          (fractal_3d < fractal_5d) & 
                          (daily_range < daily_range.rolling(10).mean()))
    
    # Early reversal detection
    reversal_signal = (fractal_break & 
                      (pressure_volume_divergence.abs() > pressure_volume_divergence.rolling(10).std()))
    
    # Trend continuation signal
    trend_continuation = (~fractal_break & 
                         (range_momentum_divergence.abs() < range_momentum_divergence.rolling(10).std()) &
                         (pressure_volume_divergence.abs() < pressure_volume_divergence.rolling(10).std()))
    
    # Combine signals with weights
    factor = (
        breakout_signal.astype(float) * 0.3 +
        accumulation_signal.astype(float) * 0.2 +
        reversal_signal.astype(float) * -0.4 +
        trend_continuation.astype(float) * 0.1 +
        range_momentum_divergence * 0.1 +
        pressure_volume_divergence * 0.1
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(20).mean()) / (factor.rolling(20).std() + 1e-8)
    
    return factor
