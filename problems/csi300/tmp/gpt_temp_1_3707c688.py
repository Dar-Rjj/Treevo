import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Transition Volume Efficiency Factor
    Combines volatility regime detection with volume efficiency analysis
    """
    data = df.copy()
    
    # Multi-Timeframe Volatility Calculation
    # Daily high-low range
    daily_range = data['high'] - data['low']
    
    # 5-day rolling range
    rolling_5d_range = daily_range.rolling(window=5, min_periods=3).mean()
    
    # 20-day volatility percentiles
    vol_percentile = daily_range.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Volatility Regime Classification
    high_vol_regime = vol_percentile > 0.7
    low_vol_regime = vol_percentile < 0.3
    normal_vol_regime = (vol_percentile >= 0.3) & (vol_percentile <= 0.7)
    
    # Regime Transition Detection
    regime_shift = pd.Series(0, index=data.index)
    
    # High-to-Normal transitions (volatility decreasing)
    high_to_normal = (high_vol_regime.shift(1)) & (normal_vol_regime)
    
    # Low-to-High transitions (volatility breakout)
    low_to_high = (low_vol_regime.shift(1)) & (high_vol_regime)
    
    # Normal regime stability (no change)
    normal_stable = (normal_vol_regime.shift(1)) & (normal_vol_regime)
    
    regime_shift[high_to_normal] = -1  # Decreasing volatility
    regime_shift[low_to_high] = 1      # Increasing volatility
    regime_shift[normal_stable] = 0.5   # Stable normal regime
    
    # Volume Expansion Analysis
    volume_3d_growth = data['volume'] / data['volume'].rolling(window=3, min_periods=2).mean() - 1
    volume_10d_baseline = data['volume'].rolling(window=10, min_periods=5).mean()
    volume_expansion = (data['volume'] > volume_10d_baseline * 1.2) & (volume_3d_growth > 0.1)
    
    # Price Response Efficiency
    intraday_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume-weighted efficiency during expansion
    volume_weighted_efficiency = intraday_efficiency * np.sqrt(data['volume'])
    
    # Transition Clarity Score
    transition_clarity = abs(regime_shift) * (1 + volume_3d_growth)
    
    # Final Factor Calculation
    factor = regime_shift * volume_weighted_efficiency * transition_clarity
    
    # Normalize by regime persistence
    regime_persistence = normal_vol_regime.rolling(window=5, min_periods=3).sum()
    factor = factor / (1 + regime_persistence)
    
    # Handle edge cases
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(0)
    
    return factor
