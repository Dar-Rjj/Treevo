import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor with regime-adaptive signals, cleaner momentum-volume alignment,
    and multi-timeframe efficiency confirmation.
    
    Economic intuition:
    - Regime-Adaptive Momentum: Momentum signals that adapt to volatility and trend regimes
    - Clean Momentum-Volume Alignment: Direct comparison of price and volume momentum trends
    - Multi-Timeframe Efficiency: Price efficiency signals across short and medium horizons
    - Dynamic Signal Weighting: Component weights adjust based on market conditions
    """
    
    # 1. Regime-adaptive momentum with dual confirmation
    # Short-term momentum (2-day)
    momentum_short = (df['close'] - df['close'].shift(2)) / (df['close'].shift(2) + 1e-7)
    
    # Medium-term momentum (5-day) for trend confirmation
    momentum_medium = (df['close'] - df['close'].shift(5)) / (df['close'].shift(5) + 1e-7)
    
    # Volatility regime detection
    price_volatility = df['close'].pct_change().rolling(10).std()
    volatility_regime = price_volatility > price_volatility.rolling(20).mean()
    
    # Trend regime detection
    trend_strength = df['close'].rolling(10).apply(lambda x: (x[-1] - x[0]) / (x.max() - x.min() + 1e-7))
    trend_regime = abs(trend_strength) > 0.3
    
    # Regime-adaptive momentum combination
    regime_momentum = np.where(
        volatility_regime & ~trend_regime,  # High vol, no clear trend
        momentum_short * 0.8 + momentum_medium * 0.2,
        np.where(
            ~volatility_regime & trend_regime,  # Low vol, clear trend
            momentum_short * 0.3 + momentum_medium * 0.7,
            momentum_short * 0.5 + momentum_medium * 0.5  # Balanced regime
        )
    )
    
    # 2. Clean momentum-volume alignment
    # Price momentum (3-day)
    price_momentum_3d = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) + 1e-7)
    
    # Volume momentum (3-day)
    volume_momentum_3d = (df['volume'] - df['volume'].shift(3)) / (df['volume'].shift(3) + 1e-7)
    
    # Momentum-volume alignment score
    momentum_volume_alignment = price_momentum_3d * volume_momentum_3d
    
    # Directional volume intensity
    price_direction = np.sign(df['close'] - df['open'])
    directional_volume_intensity = (df['volume'] * price_direction) / (df['volume'].rolling(5).mean() + 1e-7)
    
    # 3. Multi-timeframe efficiency confirmation
    # Short-term efficiency (daily)
    daily_efficiency = (df['close'] - df['open']) / ((df['high'] - df['low']) + 1e-7)
    
    # Medium-term efficiency (3-day range)
    three_day_high = df['high'].rolling(3).max()
    three_day_low = df['low'].rolling(3).min()
    three_day_efficiency = (df['close'] - df['close'].shift(2)) / ((three_day_high - three_day_low) + 1e-7)
    
    # Efficiency trend confirmation
    efficiency_trend = daily_efficiency.rolling(3).mean() - daily_efficiency.rolling(10).mean()
    
    # Combined efficiency score
    combined_efficiency = daily_efficiency * 0.6 + three_day_efficiency * 0.3 + efficiency_trend * 0.1
    
    # 4. Dynamic signal weighting based on regimes
    # Higher momentum weight in trending markets
    momentum_weight = np.where(trend_regime, 0.4, 0.3)
    
    # Higher volume alignment weight in volatile markets
    volume_weight = np.where(volatility_regime, 0.35, 0.25)
    
    # Efficiency weight adjusts inversely
    efficiency_weight = 1.0 - momentum_weight - volume_weight
    
    # Final alpha factor with dynamic weighting
    alpha_factor = (
        regime_momentum * momentum_weight +
        (momentum_volume_alignment + directional_volume_intensity) * 0.5 * volume_weight +
        combined_efficiency * efficiency_weight
    )
    
    return alpha_factor
