import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Simplified alpha factor focusing on momentum acceleration, volume confirmation, 
    and volatility efficiency with dynamic regime-aware weighting.
    
    Interpretation:
    - Momentum acceleration captures trend changes across short and medium horizons
    - Volume confirmation validates momentum signals with regime awareness
    - Volatility efficiency measures price movement quality relative to trading range
    - Dynamic weights adapt to market conditions (trending vs choppy)
    """
    
    # 1. Core momentum signals
    # Short-term momentum (5-day)
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Medium-term momentum (10-day)  
    mom_10d = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Momentum acceleration - rate of change between horizons
    momentum_accel = mom_10d - mom_5d
    
    # 2. Volume confirmation with regime detection
    # Volume momentum relative to recent average
    vol_ratio = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    
    # Volume confirmation aligned with momentum direction
    volume_confirm = vol_ratio * np.sign(mom_5d)
    
    # 3. Volatility efficiency
    # True range calculation
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Daily return efficiency within trading range
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    range_efficiency = abs(daily_return) / (true_range + 1e-7)
    
    # 4. Dynamic regime detection for weighting
    # Trend strength using rolling correlation
    trend_strength = df['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.corrcoef(np.arange(len(x)), x)[0,1] if len(x) > 1 else 0
    )
    
    # Regime-aware dynamic weights
    # Higher momentum weight in strong trends, higher efficiency weight in choppy markets
    momentum_weight = 0.5 + 0.3 * abs(trend_strength)
    volume_weight = 0.3
    efficiency_weight = 0.5 - 0.3 * abs(trend_strength)
    
    # 5. Final alpha factor combination
    alpha_factor = (
        momentum_weight * momentum_accel +
        volume_weight * volume_confirm +
        efficiency_weight * range_efficiency
    )
    
    return alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
