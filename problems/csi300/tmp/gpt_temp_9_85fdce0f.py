import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-resolution momentum convergence with adaptive volatility regimes and volume-flow efficiency
    # Combines short-term momentum acceleration, medium-term trend confirmation, and volume-flow dynamics
    
    # Multi-resolution momentum (3-day, 8-day, 13-day) with exponential weighting
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8) 
    momentum_13d = (df['close'] - df['close'].shift(13)) / df['close'].shift(13)
    
    # Momentum convergence with exponential decay weights
    momentum_convergence = (2.5 * momentum_3d + 1.2 * momentum_8d - 0.3 * momentum_13d)
    
    # Adaptive volatility regime detection using multi-scale ATR
    true_range = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Volatility regime ratio with adaptive smoothing
    short_term_vol = true_range.rolling(window=3, min_periods=1).mean()
    medium_term_vol = true_range.rolling(window=10, min_periods=1).mean()
    vol_regime_ratio = short_term_vol / (medium_term_vol + 1e-7)
    
    # Nonlinear volatility regime adjustment with adaptive scaling
    volatility_component = np.tanh((vol_regime_ratio - 1) * 2) * np.log1p(abs(vol_regime_ratio - 1))
    
    # Volume-flow efficiency with momentum alignment
    volume_trend = df['volume'].rolling(window=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan,
        raw=False
    )
    
    # Price efficiency with intraday momentum
    intraday_range = df['high'] - df['low']
    price_efficiency = (df['close'] - df['open']) / (intraday_range + 1e-7)
    
    # Volume-flow alignment with price momentum
    volume_momentum_alignment = np.sign(volume_trend) * np.sign(price_efficiency)
    volume_flow_component = np.arctan(volume_trend * 0.01) * volume_momentum_alignment * np.sqrt(abs(price_efficiency))
    
    # Nonlinear combination with regime adaptation
    momentum_enhanced = np.sign(momentum_convergence) * np.log1p(abs(momentum_convergence * 10))
    regime_adaptive = 1 + 0.8 * volatility_component  # Enhanced regime sensitivity
    volume_bounded = np.tanh(volume_flow_component * 3)  # Controlled volume contribution
    
    # Final alpha: regime-adaptive momentum-volume convergence
    alpha_factor = momentum_enhanced * regime_adaptive * volume_bounded
    
    return alpha_factor
