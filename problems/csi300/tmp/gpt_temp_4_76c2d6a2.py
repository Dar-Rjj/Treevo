import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and percentile regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, multi-day) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Percentile-based regime classification for robust market state identification
    - Multiplicative combination of momentum and volume components for enhanced signal strength
    - Dynamic weighting based on momentum consistency and volume confirmation
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Hierarchical momentum components
    intraday_return = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    daily_return = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    multi_day_momentum = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = intraday_return - overnight_return
    short_term_accel = daily_return - intraday_return
    medium_term_accel = multi_day_momentum - daily_return
    
    # Volume divergence components
    volume_ratio = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    amount_ratio = df['amount'] / (df['amount'].rolling(window=5).mean() + 1e-7)
    volume_amount_divergence = volume_ratio - amount_ratio
    
    # Percentile-based regime classification
    momentum_strength = (intraday_return.abs() + daily_return.abs() + multi_day_momentum.abs()) / 3
    momentum_strength_pct = momentum_strength.rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + (x.iloc[-1] > np.percentile(x, 30)) * 1, raw=False)
    
    volume_strength = (volume_ratio + amount_ratio) / 2
    volume_strength_pct = volume_strength.rolling(window=20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 70)) * 2 + (x.iloc[-1] > np.percentile(x, 30)) * 1, raw=False)
    
    # Multiplicative combinations
    momentum_volume_sync = intraday_return * volume_ratio * np.sign(intraday_return)
    acceleration_volume_sync = ultra_short_accel * volume_ratio * np.sign(ultra_short_accel)
    divergence_momentum = volume_amount_divergence * daily_return * np.sign(volume_amount_divergence * daily_return)
    
    # Hierarchical timeframe weighting
    ultra_short_component = ultra_short_accel * momentum_strength_pct
    short_term_component = short_term_accel * (momentum_strength_pct + volume_strength_pct)
    medium_term_component = medium_term_accel * volume_strength_pct
    
    # Volume divergence weighting
    volume_confirmation = momentum_volume_sync * volume_strength_pct
    acceleration_confirmation = acceleration_volume_sync * (momentum_strength_pct + volume_strength_pct)
    divergence_component = divergence_momentum * np.abs(volume_amount_divergence)
    
    # Combined alpha factor with hierarchical structure
    alpha_factor = (
        ultra_short_component * 0.25 +
        short_term_component * 0.35 +
        medium_term_component * 0.20 +
        volume_confirmation * 0.10 +
        acceleration_confirmation * 0.15 +
        divergence_component * 0.05
    )
    
    return alpha_factor
