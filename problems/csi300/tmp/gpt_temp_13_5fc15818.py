import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility computation
    returns = df['close'].pct_change()
    
    # Volatility Regime Detection
    vol_10d = returns.rolling(window=10, min_periods=5).std()
    vol_median_50d = vol_10d.rolling(window=50, min_periods=25).median()
    volatility_regime = (vol_10d > vol_median_50d).astype(int)
    
    # Gap Analysis with Regime Adaptation
    gap_magnitude = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # High volatility regime: focus on large gaps with volatility scaling
    large_gaps = gap_magnitude.abs()
    high_vol_reaction = np.where(
        gap_magnitude > 0,
        large_gaps * (1 + vol_10d),
        -large_gaps * (1 + vol_10d)
    )
    
    # Low volatility regime: focus on moderate gaps with momentum alignment
    momentum_5d = df['close'].pct_change(5)
    moderate_gaps = gap_magnitude.clip(lower=-0.03, upper=0.03)
    low_vol_reaction = np.where(
        gap_magnitude * momentum_5d > 0,
        moderate_gaps * (1 + momentum_5d.abs()),
        -moderate_gaps * (1 + momentum_5d.abs())
    )
    
    # Combine regime-adaptive gap reactions
    gap_reaction = np.where(
        volatility_regime == 1,
        high_vol_reaction,
        low_vol_reaction
    )
    
    # Fractal Volume Confirmation
    # Multi-scale volume ratios
    vol_5d = df['volume'].rolling(window=5, min_periods=3).mean()
    vol_20d = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = vol_5d / vol_20d
    
    # Volume clustering intensity
    volume_spike_threshold = df['volume'].rolling(window=20, min_periods=10).quantile(0.8)
    volume_spikes = (df['volume'] > volume_spike_threshold).astype(int)
    spike_count_5d = volume_spikes.rolling(window=5, min_periods=3).sum()
    
    # Volume autocorrelation (1-day lag)
    volume_autocorr = df['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    ).fillna(0)
    
    # Combine volume confirmation signals
    volume_confirmation = volume_ratio * (1 + spike_count_5d/5) * (1 + volume_autocorr.abs())
    
    # Adaptive Alpha Construction
    # Combine gap reaction with volume confirmation
    base_alpha = gap_reaction * volume_confirmation
    
    # Regime persistence multiplier
    regime_persistence = volatility_regime.rolling(window=5, min_periods=3).mean()
    persistence_multiplier = 1 + regime_persistence * 0.5
    
    # Final alpha with scaling
    alpha = base_alpha * persistence_multiplier * gap_magnitude.abs() * (1 + spike_count_5d/5)
    
    return pd.Series(alpha, index=df.index)
