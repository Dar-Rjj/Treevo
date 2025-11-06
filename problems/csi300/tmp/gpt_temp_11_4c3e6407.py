import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Regime Adaptive Alpha factor that adapts to different volatility regimes
    and applies regime-specific fractal-based factors.
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Multi-scale Volatility Measurement
    vol_short = returns.rolling(window=5).std()
    vol_medium = returns.rolling(window=10).std()
    vol_long = returns.rolling(window=20).std()
    
    # Fractal Regime Classification
    high_vol_regime = vol_short > vol_medium
    low_vol_regime = vol_short < vol_long
    transition_regime = ~high_vol_regime & ~low_vol_regime
    
    # Calculate fractal-based features
    # Price path complexity (Hurst-like approximation)
    def hurst_approximation(price_series, window=10):
        lags = range(2, window)
        tau = []
        for lag in lags:
            ts = price_series.diff(lag).dropna()
            tau.append(np.sqrt(np.mean(ts**2)))
        if len(tau) > 1:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0
        return 1.0
    
    hurst_values = df['close'].rolling(window=20).apply(
        lambda x: hurst_approximation(pd.Series(x)), raw=False
    )
    
    # Volume fractal dimension approximation
    def volume_fractal(volume_series, window=10):
        if len(volume_series) < window:
            return 1.0
        log_range = np.log(np.max(volume_series) - np.min(volume_series) + 1e-8)
        log_window = np.log(window)
        return log_range / log_window if log_window > 0 else 1.0
    
    volume_fractal_values = df['volume'].rolling(window=10).apply(
        lambda x: volume_fractal(pd.Series(x)), raw=False
    )
    
    # Price range compression
    daily_range = (df['high'] - df['low']) / df['close']
    range_compression = daily_range.rolling(window=5).std()
    
    # Volume acceleration
    volume_ma = df['volume'].rolling(window=5).mean()
    volume_accel = df['volume'] / volume_ma - 1
    
    # Multi-scale momentum
    mom_short = df['close'] / df['close'].shift(3) - 1
    mom_medium = df['close'] / df['close'].shift(8) - 1
    mom_long = df['close'] / df['close'].shift(15) - 1
    
    # Mean reversion strength
    price_zscore = (df['close'] - df['close'].rolling(window=10).mean()) / df['close'].rolling(window=10).std()
    
    # Regime-specific factor calculation
    factor = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime factors
    high_vol_factor = (
        -price_zscore * 0.4 +  # Mean reversion
        -volume_accel * 0.3 +  # Volume spike reversion
        (hurst_values - 0.5) * 0.3  # Path complexity
    )
    
    # Transition regime factors
    transition_factor = (
        (mom_short - mom_medium) * 0.4 +  # Momentum convergence
        volume_fractal_values * 0.3 +  # Volume fractal transitions
        (vol_medium - vol_short) * 0.3  # Volatility regime change
    )
    
    # Low volatility regime factors
    low_vol_factor = (
        mom_medium * 0.4 +  # Medium-term momentum
        -range_compression * 0.3 +  # Breakout potential from compression
        volume_fractal_values * 0.3  # Accumulation patterns
    )
    
    # Combine regimes with smoothing
    factor[high_vol_regime] = high_vol_factor[high_vol_regime]
    factor[transition_regime] = transition_factor[transition_regime]
    factor[low_vol_regime] = low_vol_factor[low_vol_regime]
    
    # Fill any NaN values with neutral factor (0)
    factor = factor.fillna(0)
    
    # Final smoothing and normalization
    factor = factor.rolling(window=3, min_periods=1).mean()
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor.fillna(0)
