import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Novel alpha factor combining price momentum, volume confirmation, 
    and volatility efficiency with clear economic rationale.
    """
    
    # Price momentum: 5-day vs 10-day momentum difference
    # Rationale: Captures acceleration in price movement
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    mom_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    momentum_acceleration = mom_5d - mom_10d
    
    # Volume confirmation: Recent volume vs historical average
    # Rationale: High volume confirms price momentum strength
    vol_ratio_5d = df['volume'] / df['volume'].rolling(window=5, min_periods=1).mean()
    vol_ratio_20d = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
    volume_confirmation = vol_ratio_5d - vol_ratio_20d
    
    # Volatility efficiency: Price movement per unit of volatility
    # Rationale: Efficient price moves (low volatility) are more sustainable
    price_range = (df['high'] - df['low']) / df['close']
    volatility_5d = price_range.rolling(window=5, min_periods=1).mean()
    volatility_efficiency = momentum_acceleration / (volatility_5d + 0.001)
    
    # Trend persistence: Consistency of price movement
    # Rationale: Persistent trends are more reliable
    daily_returns = df['close'].pct_change()
    trend_persistence = daily_returns.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(np.sign(x)) / len(x) if len(x) > 0 else 0
    )
    
    # Clean factor combination with clear economic interpretation:
    # Momentum acceleration amplified by volume confirmation,
    # adjusted for volatility efficiency, and weighted by trend persistence
    alpha_factor = (
        momentum_acceleration * 
        volume_confirmation * 
        volatility_efficiency * 
        trend_persistence
    )
    
    return alpha_factor
