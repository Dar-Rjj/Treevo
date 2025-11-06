import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df = df.copy()
    high = df['high']
    low = df['low']
    close_prev = df['close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close_prev)
    tr3 = abs(low - close_prev)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate volatility regimes
    vol_10 = true_range.rolling(window=10, min_periods=5).mean()
    vol_20 = true_range.rolling(window=20, min_periods=10).mean()
    
    # Define volatility regimes
    low_vol_regime = (vol_10 < vol_10.rolling(window=50, min_periods=25).quantile(0.3)) & (vol_20 < vol_20.rolling(window=50, min_periods=25).quantile(0.3))
    high_vol_regime = (vol_10 > vol_10.rolling(window=50, min_periods=25).quantile(0.7)) & (vol_20 > vol_20.rolling(window=50, min_periods=25).quantile(0.7))
    normal_vol_regime = ~low_vol_regime & ~high_vol_regime
    
    # Calculate volume percentiles within each regime
    volume = df['volume']
    
    # Volume percentile within low volatility regime
    low_vol_volume_percentile = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 50 and low_vol_regime.iloc[i]:
            window_data = volume.iloc[max(0, i-49):i+1]
            regime_mask = low_vol_regime.iloc[max(0, i-49):i+1]
            regime_volumes = window_data[regime_mask]
            if len(regime_volumes) > 5:
                low_vol_volume_percentile.iloc[i] = (volume.iloc[i] > regime_volumes.quantile(0.7))
    
    # Volume percentile within high volatility regime
    high_vol_volume_percentile = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 50 and high_vol_regime.iloc[i]:
            window_data = volume.iloc[max(0, i-49):i+1]
            regime_mask = high_vol_regime.iloc[max(0, i-49):i+1]
            regime_volumes = window_data[regime_mask]
            if len(regime_volumes) > 5:
                high_vol_volume_percentile.iloc[i] = (volume.iloc[i] < regime_volumes.quantile(0.3))
    
    # Detect volatility breakouts
    vol_breakout = (true_range > vol_20 * 1.5) & (true_range > true_range.rolling(window=20, min_periods=10).quantile(0.8))
    
    # Volume spikes during transitions
    volume_spike = (volume > volume.rolling(window=20, min_periods=10).mean() * 1.5)
    
    # Generate alpha signals
    alpha_signal = pd.Series(0.0, index=df.index)
    
    # Low Volatility Breakout Signal (High volume breakout)
    low_vol_breakout = low_vol_regime & vol_breakout & (low_vol_volume_percentile.fillna(0) > 0)
    alpha_signal[low_vol_breakout] += 1.0
    
    # High Volatility Exhaustion Signal (Low volume after spike)
    high_vol_exhaustion = high_vol_regime & ~vol_breakout & (high_vol_volume_percentile.fillna(0) > 0)
    alpha_signal[high_vol_exhaustion] -= 1.0
    
    # Regime Confirmation Signal (Volume confirms volatility trend)
    regime_confirmation = normal_vol_regime & volume_spike & (true_range > true_range.rolling(window=10, min_periods=5).mean())
    alpha_signal[regime_confirmation] += 0.5
    
    # Early regime change indicators
    early_low_to_normal = low_vol_regime.shift(1) & normal_vol_regime & volume_spike
    early_high_to_normal = high_vol_regime.shift(1) & normal_vol_regime & ~volume_spike
    
    alpha_signal[early_low_to_normal] += 0.3
    alpha_signal[early_high_to_normal] += 0.3
    
    # Smooth the signal with a 3-day moving average
    alpha_signal = alpha_signal.rolling(window=3, min_periods=1).mean()
    
    return alpha_signal
