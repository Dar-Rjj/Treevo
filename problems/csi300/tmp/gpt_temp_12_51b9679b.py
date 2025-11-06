import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum-volume composite with dual efficiency penalties and 
    regime-adaptive volatility scaling. Combines short (3-day), medium (8-day), 
    and long (15-day) momentum signals weighted by volume confirmation. Applies 
    efficiency penalties for gap absorption and intraday range utilization, 
    normalized by momentum-regime volatility scaling.
    
    Interpretable as: Stocks with persistent momentum across multiple timeframes, 
    volume-supported price moves, efficient gap closure and range utilization, 
    adjusted for current market volatility regime.
    """
    # Multi-timeframe price momentum (3d, 8d, 15d)
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    momentum_8d = df['close'] / df['close'].shift(8) - 1
    momentum_15d = df['close'] / df['close'].shift(15) - 1
    
    # Volume confirmation aligned with momentum direction
    volume_conf_3d = (df['volume'] / df['volume'].shift(3) - 1) * np.sign(momentum_3d)
    volume_conf_8d = (df['volume'] / df['volume'].shift(8) - 1) * np.sign(momentum_8d)
    volume_conf_15d = (df['volume'] / df['volume'].shift(15) - 1) * np.sign(momentum_15d)
    
    # Weighted momentum-volume composite (decaying weights for longer timeframes)
    momentum_volume_composite = (0.6 * momentum_3d * (1 + volume_conf_3d) + 
                               0.3 * momentum_8d * (1 + volume_conf_8d) + 
                               0.1 * momentum_15d * (1 + volume_conf_15d))
    
    # Gap absorption efficiency penalty
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_range = (df['high'] - df['low']) / df['close']
    gap_efficiency = 1 - (abs(opening_gap) / (daily_range + 1e-7))
    
    # Intraday range utilization penalty
    price_movement = abs(df['close'] - df['open'])
    range_utilization = price_movement / (daily_range * df['close'] + 1e-7)
    range_utilization_penalty = 1 - range_utilization
    
    # Multi-timeframe volatility measures
    volatility_3d = momentum_3d.rolling(window=6, min_periods=4).std()
    volatility_8d = momentum_8d.rolling(window=12, min_periods=8).std()
    volatility_15d = momentum_15d.rolling(window=18, min_periods=12).std()
    
    # Regime-adaptive volatility scaling based on recent momentum persistence
    momentum_persistence_3d = momentum_3d.rolling(window=6).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 6 else np.nan)
    momentum_persistence_8d = momentum_8d.rolling(window=12).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 12 else np.nan)
    momentum_persistence_15d = momentum_15d.rolling(window=18).apply(lambda x: (x > 0).sum() / len(x) if len(x) == 18 else np.nan)
    
    total_persistence = (momentum_persistence_3d + momentum_persistence_8d + momentum_persistence_15d + 1e-7)
    regime_volatility = (momentum_persistence_3d * volatility_3d + 
                        momentum_persistence_8d * volatility_8d + 
                        momentum_persistence_15d * volatility_15d) / total_persistence
    
    # Composite factor: [Momentum-volume × efficiency penalties] / regime volatility
    alpha_factor = (momentum_volume_composite * gap_efficiency * range_utilization_penalty) / (regime_volatility + 1e-7)
    
    return alpha_factor
