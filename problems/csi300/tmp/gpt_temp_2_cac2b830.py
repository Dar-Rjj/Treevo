import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Regime-Volume Composite Factor combining price pressure regimes
    with multi-timeframe volume dynamics for regime-adaptive signal generation.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Intraday Price Pressure Regime
    # Calculate Intraday Strength
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Regime Classification
    up_regime = intraday_strength > 0.3
    down_regime = intraday_strength < -0.3
    neutral_regime = ~(up_regime | down_regime)
    
    # 2. Multi-Timeframe Volume Dynamics
    # Short-term Volume Acceleration
    volume_ratio = data['volume'] / data['volume'].shift(1)
    avg_volume_ratio_5d = volume_ratio.rolling(window=5).mean()
    short_term_accel = volume_ratio / (avg_volume_ratio_5d + 1e-8)
    
    # Medium-term Volume Trend
    volume_ratio_5d = data['volume'] / data['volume'].shift(5)
    volume_trend_20d = data['volume'].rolling(window=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
    )
    medium_term_trend = volume_ratio_5d / (np.abs(volume_trend_20d) + 1e-8)
    
    # 3. Regime-Adaptive Signal Generation
    factor = pd.Series(index=data.index, dtype=float)
    
    # Up Regime Factor
    up_factor = intraday_strength * short_term_accel * medium_term_trend
    factor[up_regime] = up_factor[up_regime]
    
    # Down Regime Factor (use deceleration for down moves)
    down_accel = 2 - short_term_accel  # Inverse relationship for down moves
    down_factor = intraday_strength * down_accel * (2 - medium_term_trend)
    factor[down_regime] = down_factor[down_regime]
    
    # Neutral Regime Factor
    prev_return = data['close'].pct_change()
    volume_confirmation = (data['volume'] > data['volume'].rolling(5).mean()).astype(int)
    neutral_factor = -prev_return * volume_confirmation * np.abs(intraday_strength)
    factor[neutral_regime] = neutral_factor[neutral_regime]
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=20).mean()) / (factor.rolling(window=20).std() + 1e-8)
    
    return factor
