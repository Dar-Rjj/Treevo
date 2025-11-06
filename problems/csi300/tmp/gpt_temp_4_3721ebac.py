import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum with robust combination
    mom_5d = df['close'] / df['close'].shift(5) - 1
    mom_10d = df['close'] / df['close'].shift(10) - 1
    mom_20d = df['close'] / df['close'].shift(20) - 1
    
    # Momentum acceleration across timeframes
    mom_accel_short = mom_5d - mom_10d
    mom_accel_long = mom_10d - mom_20d
    
    # Robust momentum combination using median
    momentum_signals = pd.concat([mom_accel_short, mom_accel_long], axis=1)
    combined_momentum = momentum_signals.apply(lambda x: np.median(x), axis=1)
    
    # Multi-timeframe volume analysis
    volume_5d_median = df['volume'].rolling(window=5, min_periods=3).median()
    volume_20d_median = df['volume'].rolling(window=20, min_periods=10).median()
    
    # Volume regime detection using percentiles
    volume_20d_p25 = df['volume'].rolling(window=20, min_periods=10).quantile(0.25)
    volume_20d_p75 = df['volume'].rolling(window=20, min_periods=10).quantile(0.75)
    
    # Volume confirmation signals
    volume_short_term = df['volume'] / (volume_5d_median + 1e-7)
    volume_regime = (df['volume'] - volume_20d_p25) / (volume_20d_p75 - volume_20d_p25 + 1e-7)
    
    # Non-linear volume transformation
    volume_signal = np.log1p(volume_short_term) * np.sqrt(np.abs(volume_regime))
    
    # Robust volatility using interquartile range
    returns = df['close'].pct_change()
    rolling_iqr = returns.rolling(window=15, min_periods=8).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )
    
    # Volatility regime detection
    vol_5d = returns.rolling(window=5, min_periods=3).std()
    vol_20d = returns.rolling(window=20, min_periods=10).std()
    vol_regime = vol_5d / (vol_20d + 1e-7)
    
    # Price efficiency with amount (dollar volume) adjustment
    range_utilization = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    amount_per_volume = df['amount'] / (df['volume'] + 1e-7)  # Average trade size
    
    # Price efficiency signal with liquidity adjustment
    efficiency_signal = range_utilization * np.log1p(amount_per_volume / df['close'])
    
    # Volatility-adjusted momentum with regime awareness
    vol_adjusted_mom = combined_momentum / (rolling_iqr + 1e-7)
    regime_adaptive_mom = vol_adjusted_mom * np.tanh(vol_regime * 2)  # Non-linear regime response
    
    # Final factor combination with non-linear interactions
    volume_confirmed_momentum = regime_adaptive_mom * volume_signal
    factor = volume_confirmed_momentum * np.power(efficiency_signal, 0.5)  # Square root transform
    
    return factor
