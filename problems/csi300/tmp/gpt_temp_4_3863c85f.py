import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor integrating multi-scale momentum, volume-price confirmation, 
    and dynamic volatility adjustment for superior return prediction.
    
    Economic intuition:
    - Multi-Scale Momentum Fusion: Combines ultra-short, short, and medium-term momentum
    - Volume-Price Confirmation: Validates price moves with volume intensity and persistence
    - Dynamic Volatility Adjustment: Adapts signals to changing market volatility regimes
    - Momentum Quality Assessment: Filters momentum signals based on price efficiency
    """
    
    # 1. Multi-scale momentum fusion
    # Combines 1-day, 3-day, and 5-day momentum for comprehensive trend capture
    momentum_ultra_short = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-7)
    momentum_short = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) + 1e-7)
    momentum_medium = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-7)
    
    # Weighted combination favoring recent momentum
    momentum_fusion = (
        momentum_ultra_short * 0.5 + 
        momentum_short * 0.3 + 
        momentum_medium * 0.2
    )
    
    # 2. Volume-price confirmation
    # Validates price momentum with volume intensity and persistence
    price_direction = np.sign(df['close'] - df['close'].shift(1))
    volume_intensity = df['volume'] / (df['volume'].rolling(3).mean() + 1e-7)
    volume_persistence = df['volume'].rolling(3).apply(lambda x: np.corrcoef(range(3), x)[0,1] if not x.isnull().any() else 0)
    
    # Volume confirmation: positive when volume supports price direction
    volume_confirmation = price_direction * volume_intensity * (1 + volume_persistence)
    
    # 3. Dynamic volatility adjustment
    # Uses adaptive volatility scaling based on recent market conditions
    price_volatility = df['close'].pct_change().rolling(5).std()
    range_volatility = ((df['high'] - df['low']) / df['close']).rolling(5).std()
    combined_volatility = (price_volatility + range_volatility) / 2
    
    # Volatility adjustment: stronger signals in low volatility, weaker in high volatility
    volatility_adjustment = 1.0 / (combined_volatility + 1e-7)
    
    # 4. Momentum quality assessment
    # Filters momentum based on price efficiency and trend consistency
    intraday_efficiency = (df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-7)
    trend_consistency = df['close'].rolling(3).apply(lambda x: np.corrcoef(range(3), x)[0,1] if not x.isnull().any() else 0)
    
    # Quality filter: rewards efficient, consistent momentum
    momentum_quality = momentum_fusion * intraday_efficiency * (1 + trend_consistency)
    
    # Strategic integration with economic rationale
    alpha_factor = (
        momentum_quality * 0.40 * volatility_adjustment +      # Quality-adjusted momentum scaled for volatility
        volume_confirmation * 0.35 +                          # Volume validation of price moves
        momentum_fusion * 0.25 * volatility_adjustment        # Raw momentum with volatility scaling
    )
    
    return alpha_factor
