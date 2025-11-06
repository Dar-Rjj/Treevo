import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Alpha Factor
    Adapts trading signals based on identified volatility regimes
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility Regime Identification
    # Calculate rolling 20-day high-low volatility
    hl_range = (df['high'] - df['low']) / df['close']
    volatility_20d = hl_range.rolling(window=20, min_periods=10).std()
    
    # Calculate historical volatility percentiles (using expanding window)
    vol_percentile = volatility_20d.expanding(min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 50 else np.nan
    )
    
    # Classify volatility regimes
    high_vol_regime = vol_percentile > 0.7
    low_vol_regime = vol_percentile < 0.3
    transition_regime = (~high_vol_regime) & (~low_vol_regime)
    
    # Price Pattern Recognition
    # Detect local minima/maxima for support/resistance
    close_5d_min = df['close'].rolling(window=5, center=True, min_periods=3).min()
    close_5d_max = df['close'].rolling(window=5, center=True, min_periods=3).max()
    
    is_local_min = (df['close'] == close_5d_min) & (df['close'].shift(1) > df['close']) & (df['close'].shift(-1) > df['close'])
    is_local_max = (df['close'] == close_5d_max) & (df['close'].shift(1) < df['close']) & (df['close'].shift(-1) < df['close'])
    
    # Consolidation range detection
    range_10d = (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()) / df['close']
    is_consolidating = range_10d < range_10d.rolling(window=30).mean()
    
    # Breakout strength (price movement relative to recent range)
    breakout_strength = (df['close'] - df['close'].shift(5)) / range_10d
    
    # Volume-Volatility Interaction
    # Volume-to-volatility ratio
    volume_avg_10d = df['volume'].rolling(window=10).mean()
    vol_to_vol_ratio = df['volume'] / (volatility_20d * volume_avg_10d + 1e-8)
    
    # Volume clustering during regime transitions
    regime_changes = (high_vol_regime.astype(int).diff() != 0) | (low_vol_regime.astype(int).diff() != 0)
    volume_spike = df['volume'] > df['volume'].rolling(window=20).quantile(0.8)
    
    # Abnormal volume detection
    volume_zscore = (df['volume'] - df['volume'].rolling(window=20).mean()) / (df['volume'].rolling(window=20).std() + 1e-8)
    abnormal_volume = volume_zscore > 2.0
    
    # Regime-Specific Factor Components
    
    # High volatility regime: momentum acceleration
    momentum_5d = df['close'] / df['close'].shift(5) - 1
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    momentum_accel = momentum_5d - momentum_10d
    
    high_vol_component = momentum_accel * np.sign(momentum_5d)
    
    # Low volatility regime: mean reversion
    ma_20d = df['close'].rolling(window=20).mean()
    mean_reversion = (ma_20d - df['close']) / (df['close'].rolling(window=20).std() + 1e-8)
    
    low_vol_component = mean_reversion
    
    # Transition periods: breakout confirmation
    volume_confirmation = (volume_spike & (breakout_strength.abs() > 0.02)).astype(float)
    transition_component = breakout_strength * volume_confirmation
    
    # Adaptive Factor Construction
    # Regime persistence weighting
    regime_persistence = pd.Series(1.0, index=df.index)
    for i in range(1, len(df)):
        if high_vol_regime.iloc[i] and high_vol_regime.iloc[i-1]:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1
        elif low_vol_regime.iloc[i] and low_vol_regime.iloc[i-1]:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1
        elif transition_regime.iloc[i] and transition_regime.iloc[i-1]:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1
    
    regime_weight = np.minimum(regime_persistence / 10.0, 1.0)
    
    # Combine regime-specific components with dynamic weights
    factor = pd.Series(0.0, index=df.index)
    
    # High volatility component
    factor[high_vol_regime] = high_vol_component[high_vol_regime] * regime_weight[high_vol_regime]
    
    # Low volatility component  
    factor[low_vol_regime] = low_vol_component[low_vol_regime] * regime_weight[low_vol_regime]
    
    # Transition component
    factor[transition_regime] = transition_component[transition_regime] * regime_weight[transition_regime]
    
    # Apply regime-dependent position sizing
    # Reduce position size in high volatility, increase in low volatility
    position_size = np.where(high_vol_regime, 0.7, 
                           np.where(low_vol_regime, 1.3, 1.0))
    
    final_factor = factor * position_size
    
    # Normalize the factor
    final_factor = (final_factor - final_factor.rolling(window=50).mean()) / (final_factor.rolling(window=50).std() + 1e-8)
    
    return final_factor
