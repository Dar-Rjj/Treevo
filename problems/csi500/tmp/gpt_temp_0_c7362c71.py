import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor combining momentum, volume dynamics, volatility structure,
    and microstructure signals with clear economic rationale.
    """
    
    # 1. Momentum component: 3-day price acceleration (rate of change of momentum)
    momentum_accel = ((df['close'].shift(1) - df['close'].shift(3)) / df['close'].shift(3)) - \
                    ((df['close'].shift(3) - df['close'].shift(6)) / df['close'].shift(6))
    
    # 2. Volume dynamics: Volume trend consistency (5-day vs 10-day volume slope)
    volume_5d_trend = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: (x[-1] - x[0]) / (x[0] + 1e-7) if len(x) == 5 else np.nan
    )
    volume_10d_trend = df['volume'].rolling(window=10, min_periods=6).apply(
        lambda x: (x[-1] - x[0]) / (x[0] + 1e-7) if len(x) == 10 else np.nan
    )
    volume_trend_alignment = np.sign(volume_5d_trend) * np.sign(volume_10d_trend) * \
                           np.minimum(np.abs(volume_5d_trend), np.abs(volume_10d_trend))
    
    # 3. Volatility structure: Volatility compression/expansion ratio
    short_term_vol = (df['high'] - df['low']).rolling(window=3, min_periods=2).std()
    medium_term_vol = (df['high'] - df['low']).rolling(window=10, min_periods=6).std()
    vol_compression_ratio = short_term_vol / (medium_term_vol + 1e-7)
    
    # 4. Microstructure: Order flow intensity (amount per volume normalized by recent average)
    trade_intensity = df['amount'] / (df['volume'] + 1e-7)
    normalized_trade_intensity = trade_intensity / trade_intensity.rolling(window=15, min_periods=8).mean()
    
    # 5. Price efficiency: Close-to-open gap persistence
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_persistence = np.sign(overnight_gap) * (overnight_gap - overnight_gap.rolling(window=5, min_periods=3).mean())
    
    # Combine factors with economic rationale:
    # - Momentum acceleration indicates trend strength
    # - Volume trend alignment confirms participation
    # - Volatility compression suggests potential breakout
    # - Trade intensity reflects institutional activity
    # - Gap persistence measures market efficiency
    
    alpha_factor = (
        momentum_accel * 
        np.tanh(volume_trend_alignment * 2) *  # Non-linear volume confirmation
        (1.0 / (vol_compression_ratio + 0.5)) *  # Favor volatility compression
        np.log1p(np.abs(normalized_trade_intensity)) * np.sign(normalized_trade_intensity) *  # Microstructure signal
        np.exp(-np.abs(gap_persistence) * 3) * np.sign(gap_persistence)  # Mean reversion of inefficient gaps
    )
    
    return alpha_factor
