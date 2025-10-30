import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-timeframe Price-Volume Convergence with Regime-Based Efficiency
    """
    # Calculate Price-Volume Convergence Divergence
    # 5-day price momentum
    price_momentum = df['close'] / df['close'].shift(5) - 1
    
    # 5-day volume momentum
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Convergence score
    convergence_score = price_momentum * volume_momentum
    
    # 20-day daily return volatility
    daily_returns = df['close'].pct_change()
    vol_20d = daily_returns.rolling(window=20, min_periods=10).std()
    
    # Adjust convergence by volatility
    convergence_adjusted = convergence_score / (vol_20d + 1e-8)
    
    # Analyze Multi-timeframe Efficiency Patterns
    # Short-term efficiency
    short_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Medium-term efficiency
    close_open_diff = (df['close'] - df['open']).rolling(window=5).sum()
    high_low_range = (df['high'] - df['low']).rolling(window=5).sum()
    medium_efficiency = close_open_diff / (high_low_range + 1e-8)
    
    # Efficiency momentum
    efficiency_momentum = short_efficiency - medium_efficiency
    
    # Efficiency regimes using 20-day efficiency percentile bands
    efficiency_20d = short_efficiency.rolling(window=20, min_periods=10)
    high_efficiency_regime = short_efficiency > efficiency_20d.quantile(0.8)
    low_efficiency_regime = short_efficiency < efficiency_20d.quantile(0.2)
    
    # Volatility regimes
    vol_percentile = vol_20d.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    high_vol_regime = vol_percentile > 0.8
    low_vol_regime = vol_percentile < 0.2
    
    # Convergence strength
    convergence_strength = convergence_adjusted.rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    strong_convergence = convergence_strength > 0.6
    weak_convergence = convergence_strength < 0.4
    
    # Generate Regime-Based Convergence Signals
    factor = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime signals
    high_vol_mask = high_vol_regime.fillna(False)
    factor[high_vol_mask & strong_convergence & (efficiency_momentum > 0)] = 2.0  # Volatile breakout
    factor[high_vol_mask & strong_convergence & (efficiency_momentum <= 0)] = -2.0  # Volatile fakeout
    factor[high_vol_mask & weak_convergence & (efficiency_momentum > 0)] = 0.5  # Choppy advance
    factor[high_vol_mask & weak_convergence & (efficiency_momentum <= 0)] = -0.5  # Choppy decline
    
    # Low volatility regime signals
    low_vol_mask = low_vol_regime.fillna(False)
    factor[low_vol_mask & strong_convergence & (efficiency_momentum > 0)] = 3.0  # Clean trend initiation
    factor[low_vol_mask & strong_convergence & (efficiency_momentum <= 0)] = -3.0  # Clean reversal
    factor[low_vol_mask & weak_convergence & (efficiency_momentum > 0)] = 1.0  # Grinding accumulation
    factor[low_vol_mask & weak_convergence & (efficiency_momentum <= 0)] = -1.0  # Grinding distribution
    
    # Fill remaining with base convergence signal
    remaining_mask = factor.isna()
    factor[remaining_mask] = convergence_adjusted[remaining_mask]
    
    return factor
