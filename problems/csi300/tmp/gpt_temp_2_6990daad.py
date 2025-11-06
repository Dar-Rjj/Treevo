import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence via percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection across multiple time horizons (daily, 3-day, 5-day)
    - Percentile-based regime classification for robust regime persistence
    - Multiplicative combinations enhance signal strength during synchronized conditions
    - Smooth transitions between regimes using exponential weighting
    - Positive values indicate bullish momentum with volume confirmation across timeframes
    - Negative values suggest bearish pressure with volume distribution divergence
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    overnight_accel = overnight_momentum - overnight_momentum.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(3)
    
    # Volume divergence across multiple timeframes
    volume_daily = df['volume'] / (df['volume'].rolling(window=5).mean() + 1e-7)
    volume_3day = df['volume'].rolling(window=3).mean() / (df['volume'].rolling(window=10).mean() + 1e-7)
    volume_5day = df['volume'].rolling(window=5).mean() / (df['volume'].rolling(window=15).mean() + 1e-7)
    
    # Percentile-based regime classification for persistence
    momentum_percentile = (intraday_momentum + overnight_momentum + weekly_momentum).rolling(window=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1
    )
    
    volume_percentile = (volume_daily + volume_3day + volume_5day).rolling(window=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] > x.quantile(0.3)) * 1
    )
    
    # Regime persistence weights
    momentum_regime_weight = momentum_percentile.rolling(window=5).mean()
    volume_regime_weight = volume_percentile.rolling(window=5).mean()
    
    # Smooth transitions using exponential weighting
    momentum_smooth = (intraday_momentum.ewm(span=3).mean() * 0.4 + 
                      overnight_momentum.ewm(span=5).mean() * 0.3 + 
                      weekly_momentum.ewm(span=8).mean() * 0.3)
    
    acceleration_smooth = (intraday_accel.ewm(span=2).mean() * 0.5 + 
                          overnight_accel.ewm(span=3).mean() * 0.3 + 
                          weekly_accel.ewm(span=5).mean() * 0.2)
    
    volume_smooth = (volume_daily.ewm(span=3).mean() * 0.4 + 
                    volume_3day.ewm(span=5).mean() * 0.35 + 
                    volume_5day.ewm(span=8).mean() * 0.25)
    
    # Multiplicative combinations for enhanced signals
    momentum_volume_sync = momentum_smooth * volume_smooth * np.sign(momentum_smooth * volume_smooth)
    accel_volume_sync = acceleration_smooth * volume_smooth * np.sign(acceleration_smooth * volume_smooth)
    
    # Regime-adaptive factor combination
    alpha_factor = (
        momentum_regime_weight * momentum_smooth * 0.4 +
        volume_regime_weight * volume_smooth * 0.3 +
        momentum_volume_sync * 0.15 +
        accel_volume_sync * 0.15
    )
    
    return alpha_factor
