import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Regime-aware momentum with volume-pressure and volatility regime detection
    # Focuses on microstructure signals and non-linear regime shifts
    
    # Short-term momentum with regime filtering (exclude extreme volatility periods)
    momentum_3d = (df['close'].shift(1) - df['close'].shift(3)) / df['close'].shift(3)
    momentum_5d = (df['close'].shift(1) - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volatility regime detection using rolling percentiles
    high_low_range = (df['high'] - df['low']) / df['close']
    vol_regime = high_low_range.rolling(window=20, min_periods=10).apply(
        lambda x: 1 if x.iloc[-1] > np.percentile(x, 70) else (-1 if x.iloc[-1] < np.percentile(x, 30) else 0)
    )
    
    # Volume pressure with acceleration (rate of change)
    volume_ma_10 = df['volume'].rolling(window=10, min_periods=5).mean()
    volume_pressure = (df['volume'] - volume_ma_10) / (volume_ma_10 + 1e-7)
    volume_accel = volume_pressure - volume_pressure.shift(3)
    
    # Trade size concentration (large vs small trades)
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    trade_size_skew = avg_trade_size / avg_trade_size.rolling(window=15, min_periods=8).mean()
    
    # Intraday strength (close relative to daily range)
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-7)
    
    # Non-linear combination with regime awareness
    # In high volatility: emphasize volume acceleration and intraday strength
    # In low volatility: emphasize momentum and trade size concentration
    # In normal volatility: balanced approach
    
    base_factor = (
        momentum_3d * np.tanh(volume_pressure * 2) + 
        momentum_5d * np.tanh(trade_size_skew)
    )
    
    regime_adjusted = base_factor * (1 + 0.3 * vol_regime * np.sign(base_factor))
    
    # Add microstructure component (intraday strength modulated by volume acceleration)
    microstructure = intraday_strength * np.tanh(volume_accel * 3)
    
    alpha_factor = regime_adjusted + 0.5 * microstructure
    
    return alpha_factor
