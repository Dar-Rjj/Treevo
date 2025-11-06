import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum-velocity synchronization with volume-pressure regimes.
    
    Interpretation:
    - Momentum velocity (rate of change) across intraday, overnight, and multi-day horizons
    - Volume-pressure regimes using smooth percentile transitions
    - Multiplicative synchronization between momentum velocity and volume pressure
    - Regime persistence detection with smooth decay factors
    - Cross-timeframe momentum convergence/divergence patterns
    - Positive values indicate synchronized bullish momentum acceleration with volume confirmation
    - Negative values suggest bearish momentum deceleration with volume-pressure divergence
    """
    
    # Momentum velocity components (rate of momentum change)
    intraday_mom = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_mom = (df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-7)
    multi_day_mom = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum velocity with smooth hyperbolic transitions
    intraday_velocity = np.tanh(intraday_mom - intraday_mom.shift(1)) * np.sqrt(np.abs(intraday_mom))
    overnight_velocity = np.tanh(overnight_mom - overnight_mom.shift(1)) * np.sqrt(np.abs(overnight_mom))
    multi_day_velocity = np.tanh(multi_day_mom - multi_day_mom.shift(2)) * np.sqrt(np.abs(multi_day_mom))
    
    # Volume pressure regimes with smooth percentile transitions
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    
    short_volume_pressure = df['volume'] / (volume_5d_avg + 1e-7)
    medium_volume_pressure = df['volume'] / (volume_20d_avg + 1e-7)
    
    # Smooth percentile-based regime classification
    short_volume_percentile = short_volume_pressure.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    medium_volume_percentile = medium_volume_pressure.rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Smooth regime weights using sigmoid transitions
    short_volume_weight = 0.8 + 1.4 * (1 / (1 + np.exp(-10 * (short_volume_percentile - 0.5))))
    medium_volume_weight = 0.7 + 1.6 * (1 / (1 + np.exp(-8 * (medium_volume_percentile - 0.5))))
    
    # Volume-momentum synchronization with multiplicative enhancement
    intraday_sync = intraday_velocity * short_volume_weight * np.sign(intraday_velocity * short_volume_pressure)
    overnight_sync = overnight_velocity * short_volume_weight * np.sign(overnight_velocity * short_volume_pressure)
    multi_day_sync = multi_day_velocity * medium_volume_weight * np.sign(multi_day_velocity * medium_volume_pressure)
    
    # Cross-timeframe momentum convergence/divergence
    short_term_convergence = intraday_sync * overnight_sync * np.sign(intraday_sync * overnight_sync)
    cross_timeframe_convergence = (intraday_sync + multi_day_sync) * np.sign(intraday_sync * multi_day_sync)
    
    # Regime persistence with smooth decay
    volume_persistence = np.tanh(
        (short_volume_percentile.diff(1).abs() < 0.2).astype(float) + 
        (medium_volume_percentile.diff(1).abs() < 0.15).astype(float)
    )
    
    momentum_persistence = np.tanh(
        (intraday_velocity.diff(1).abs() < 0.1).astype(float) + 
        (multi_day_velocity.diff(2).abs() < 0.08).astype(float)
    )
    
    # Multiplicative combination hierarchy
    primary_component = (
        intraday_sync * 0.35 + 
        multi_day_sync * 0.4 + 
        short_term_convergence * 0.25
    )
    
    secondary_component = (
        cross_timeframe_convergence * 0.6 + 
        overnight_sync * 0.4
    )
    
    # Final alpha factor with regime persistence enhancement
    alpha_factor = (
        primary_component * secondary_component * np.sign(primary_component * secondary_component) * 0.7 +
        (volume_persistence * momentum_persistence) * 0.3
    ) * (1 + 0.2 * volume_persistence * momentum_persistence)
    
    return alpha_factor
