import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Multi-timeframe momentum calculation
    momentum_short = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    momentum_medium = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Momentum acceleration (change in momentum)
    momentum_acceleration = momentum_short - momentum_medium
    
    # Volatility normalization using high-low range
    daily_range = df['high'] - df['low']
    volatility = daily_range.rolling(window=20).std()
    
    # Risk-adjusted momentum acceleration
    risk_adjusted_acceleration = momentum_acceleration / (volatility + 1e-7)
    
    # Volume trend and confirmation
    volume_trend = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    # Volume confirmation: stronger when volume confirms price direction
    volume_confirmation = np.where(
        momentum_short * volume_trend > 0,
        np.abs(volume_trend) * 2,  # Amplify when confirmed
        np.abs(volume_trend) * 0.5  # Reduce when not confirmed
    )
    
    # Regime detection using volatility percentile
    volatility_rolling = volatility.rolling(window=20)
    volatility_percentile = volatility_rolling.apply(lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(int), raw=False)
    
    # Regime-aware weighting: prefer low volatility environments
    regime_weight = 1.5 - volatility_percentile  # 1.5 in low vol, 0.5 in high vol
    
    # Combine components
    raw_factor = risk_adjusted_acceleration * (1 + volume_confirmation) * regime_weight
    
    # Multi-timeframe smoothing
    factor_smooth_3 = raw_factor.ewm(span=3).mean()
    factor_smooth_5 = raw_factor.ewm(span=5).mean()
    
    # Final factor: average of smoothed versions
    final_factor = (factor_smooth_3 + factor_smooth_5) / 2
    
    return final_factor
