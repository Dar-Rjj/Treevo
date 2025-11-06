import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor combining momentum acceleration dynamics, volatility regime transitions,
    and microstructure efficiency signals for enhanced return prediction.
    
    Economic rationale:
    - Momentum acceleration captures the rate of change in price momentum, identifying stocks with
      accelerating/decelerating trends that often precede reversals or continuations
    - Volatility regime transitions identify periods when volatility patterns are changing, which
      often signal regime shifts and create predictable return patterns
    - Microstructure efficiency measures the quality of price discovery and trading activity,
      providing confirmation of the sustainability of price moves
    """
    
    # Momentum acceleration dynamics
    # Multi-timeframe momentum acceleration to capture different trend phases
    short_mom = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_mom = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    long_mom = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    
    # Acceleration as second derivative of momentum
    short_accel = short_mom - short_mom.shift(2)
    medium_accel = medium_mom - medium_mom.shift(5)
    long_accel = long_mom - long_mom.shift(8)
    
    # Combined momentum acceleration with time decay weighting
    momentum_acceleration = (
        0.5 * short_accel + 
        0.3 * medium_accel + 
        0.2 * long_accel
    )
    
    # Volatility regime transition signals
    # Multi-scale volatility ratio to detect regime transitions
    micro_vol = (df['high'] - df['low']).rolling(window=3, min_periods=1).std()
    short_vol = df['close'].pct_change().rolling(window=5, min_periods=1).std()
    medium_vol = df['close'].pct_change().rolling(window=15, min_periods=1).std()
    
    # Volatility compression/expansion signals
    vol_compression_3_5 = micro_vol / (short_vol + 1e-7)
    vol_expansion_5_15 = short_vol / (medium_vol + 1e-7)
    
    # Regime transition score - favors periods of changing volatility patterns
    regime_transition = (
        np.exp(-np.abs(vol_compression_3_5 - 1)) * 
        np.exp(-np.abs(vol_expansion_5_15 - 1))
    )
    
    # Microstructure efficiency signals
    # Price impact quality: efficiency of trading relative to price range
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    normalized_trade_impact = avg_trade_size / ((df['high'] - df['low']) + 1e-7)
    
    # Order flow persistence: consistency of trading activity
    volume_trend = df['volume'].rolling(window=5, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else 0
    )
    
    # Price discovery efficiency: how efficiently price moves within daily range
    open_to_close_efficiency = (df['close'] - df['open']).abs() / ((df['high'] - df['low']) + 1e-7)
    
    # Combined microstructure efficiency score
    microstructure_efficiency = (
        np.tanh(normalized_trade_impact / (normalized_trade_impact.rolling(window=10, min_periods=1).std() + 1e-7)) *
        np.sign(volume_trend) *
        open_to_close_efficiency
    )
    
    # Final alpha factor with economic interpretation:
    # Strong momentum acceleration during volatility regime transitions,
    # confirmed by efficient microstructure signals
    alpha_factor = (
        momentum_acceleration * 
        regime_transition * 
        microstructure_efficiency
    )
    
    return alpha_factor
