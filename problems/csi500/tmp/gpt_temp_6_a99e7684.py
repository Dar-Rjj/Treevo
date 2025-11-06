import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Advanced alpha factor with aligned 15-day periods, regime-aware volatility scaling, 
    and sophisticated signal transformations.
    
    Economic rationale:
    - 15-day price momentum provides cleaner trend signals with reduced noise
    - Volume velocity captures momentum acceleration with directional emphasis
    - Regime-adaptive volatility scaling using rolling percentiles for robustness
    - Range dominance measures intraday buyer/seller control with directional bias
    - Amount flow intensity identifies institutional participation with magnitude scaling
    - Combined: Captures sustainable medium-term trends with volume confirmation,
      adjusted for volatility regimes and enhanced by microstructure signals
    """
    
    # Aligned 15-day price momentum for cleaner trend signals
    price_momentum = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    
    # Volume velocity with directional emphasis
    volume_velocity = (df['volume'] - df['volume'].rolling(window=15, min_periods=1).mean()) / df['volume'].rolling(window=15, min_periods=1).std()
    volume_direction = np.sign(volume_velocity)
    volume_intensity = np.log1p(np.abs(volume_velocity))
    
    # Regime-adaptive volatility scaling using rolling percentiles
    daily_volatility = (df['high'] - df['low']) / df['close']
    vol_regime_adjuster = daily_volatility.rolling(window=15, min_periods=1).apply(
        lambda x: 1 / (np.percentile(x, 70) + 0.001) if len(x) == 15 else 1.0
    )
    
    # Range dominance with directional bias
    range_position = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    range_dominance = np.tanh(range_position * 3)  # Enhanced directional sensitivity
    
    # Amount flow intensity with magnitude scaling
    amount_flow = df['amount'] / df['amount'].rolling(window=15, min_periods=1).mean() - 1
    flow_direction = np.sign(amount_flow)
    flow_magnitude = np.sqrt(np.abs(amount_flow) + 1)
    
    # Sophisticated combination with regime awareness and signal transformations
    alpha_factor = (
        price_momentum * 
        volume_direction * volume_intensity *
        vol_regime_adjuster *
        range_dominance *
        flow_direction * flow_magnitude
    )
    
    return alpha_factor
