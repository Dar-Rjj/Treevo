import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor combining volatility-scaled momentum, volume-price divergence,
    and price efficiency with dynamic regime-aware weighting.
    
    Economic intuition:
    - Volatility-scaled momentum: Momentum signals adjusted for current volatility regime
    - Volume-price efficiency: Measures how efficiently volume translates to price movement
    - Price range efficiency: Captures intraday price efficiency relative to volatility
    - Dynamic regime weighting: Adapts component importance based on rolling volatility regime
    """
    
    # Volatility-scaled momentum across multiple horizons
    momentum_1d = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-7)
    momentum_2d = (df['close'] - df['close'].shift(2)) / (df['close'].shift(2) + 1e-7)
    momentum_3d = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) + 1e-7)
    
    # Rolling volatility using 5-day window for regime detection
    daily_range = (df['high'] - df['low']) / (df['close'] + 1e-7)
    rolling_volatility = daily_range.rolling(5).mean()
    
    # Volatility-scaled momentum components
    vol_scaled_momentum_1d = momentum_1d / (rolling_volatility + 1e-7)
    vol_scaled_momentum_2d = momentum_2d / (rolling_volatility + 1e-7)
    vol_scaled_momentum_3d = momentum_3d / (rolling_volatility + 1e-7)
    
    # Volume-price efficiency: measures how efficiently volume translates to price movement
    price_change_magnitude = (df['close'] - df['close'].shift(1)).abs() / (df['close'].shift(1) + 1e-7)
    volume_efficiency = df['volume'] / (df['volume'].rolling(5).mean() + 1e-7)
    volume_price_efficiency = price_change_magnitude * volume_efficiency
    
    # Price range efficiency: intraday price movement efficiency relative to volatility
    realized_range = (df['close'] - df['open']) / (df['open'] + 1e-7)
    potential_range = (df['high'] - df['low']) / (df['low'] + 1e-7)
    range_efficiency = realized_range.abs() / (potential_range + 1e-7)
    
    # Dynamic regime-aware weighting based on rolling volatility
    volatility_quantile = rolling_volatility.rolling(10).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
    
    # High volatility regime gives more weight to volatility-scaled components
    momentum_weight = np.where(volatility_quantile > 0.7, 0.6, 
                              np.where(volatility_quantile < 0.3, 0.4, 0.5))
    
    efficiency_weight = np.where(volatility_quantile > 0.7, 0.2,
                               np.where(volatility_quantile < 0.3, 0.4, 0.3))
    
    range_weight = 1 - momentum_weight - efficiency_weight
    
    # Strategic combination with dynamic regime-aware weights
    alpha_factor = (
        vol_scaled_momentum_1d * (0.4 * momentum_weight) +
        vol_scaled_momentum_2d * (0.3 * momentum_weight) +
        vol_scaled_momentum_3d * (0.3 * momentum_weight) +
        volume_price_efficiency * efficiency_weight +
        range_efficiency * range_weight
    )
    
    return alpha_factor
