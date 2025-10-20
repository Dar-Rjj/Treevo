import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Convergence Momentum
    # Velocity Momentum
    intraday_velocity = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    velocity_momentum = intraday_velocity.rolling(window=5).mean()
    
    # Efficiency Momentum
    effective_volume = df['amount'] / df['volume'].replace(0, np.nan)
    efficiency_momentum = effective_volume.rolling(window=5).mean()
    
    # Convergence Signal
    convergence_signal = velocity_momentum - efficiency_momentum
    
    # Volatility-Adaptive Momentum
    # Multi-timeframe Returns
    returns_1d = df['close'].pct_change(1)
    returns_3d = df['close'].pct_change(3)
    returns_5d = df['close'].pct_change(5)
    
    # Volatility regime
    daily_range = (df['high'] - df['low']) / df['close']
    volatility_regime = daily_range.rolling(window=20).mean()
    vol_weight_1d = returns_1d / volatility_regime.replace(0, np.nan)
    vol_weight_3d = returns_3d / volatility_regime.replace(0, np.nan)
    vol_weight_5d = returns_5d / volatility_regime.replace(0, np.nan)
    
    # Gap Momentum Persistence
    overnight_gap = (df['open'] / df['close'].shift(1) - 1).fillna(0)
    gap_filling_ratio = np.abs((df['close'] - df['open']) / (df['open'] - df['close'].shift(1)).replace(0, np.nan)).fillna(0)
    
    # Gap persistence
    gap_direction = np.sign(overnight_gap)
    consecutive_gaps = gap_direction.rolling(window=3).apply(lambda x: len(set(x)) if len(x) == 3 else np.nan)
    gap_magnitude_consistency = overnight_gap.rolling(window=5).std()
    
    # Volume-Intensity Momentum
    volume_intensity = df['volume'].rolling(window=5).apply(lambda x: x[-1] / x.mean() if x.mean() > 0 else np.nan)
    trade_size_momentum = (df['amount'] / df['volume']).pct_change(3).fillna(0)
    
    # Intensity-weighted price changes
    intensity_weighted_returns = returns_1d * volume_intensity.fillna(1)
    
    # Range-Expansion Cycles
    range_expansion_ratio = (df['high'] - df['low']).rolling(window=5).apply(
        lambda x: x[-1] / x[:-1].mean() if len(x) == 5 and x[:-1].mean() > 0 else np.nan
    )
    
    # Breakout persistence
    breakout_signal = (df['close'] > df['high'].shift(1)).astype(int) - (df['close'] < df['low'].shift(1)).astype(int)
    breakout_persistence = breakout_signal.rolling(window=5).sum()
    
    # Combine factors with appropriate weights
    factor = (
        0.25 * convergence_signal.fillna(0) +
        0.20 * (0.4 * vol_weight_1d.fillna(0) + 0.35 * vol_weight_3d.fillna(0) + 0.25 * vol_weight_5d.fillna(0)) +
        0.15 * (overnight_gap * (1 - gap_filling_ratio) * (1 / (gap_magnitude_consistency.replace(0, np.nan) + 0.01))).fillna(0) +
        0.20 * intensity_weighted_returns.fillna(0) +
        0.20 * (range_expansion_ratio.fillna(0) * breakout_persistence.fillna(0))
    )
    
    return factor
