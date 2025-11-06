import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with percentile normalization
    - Volume divergence detection across different momentum regimes
    - Dynamic regime classification based on volatility and volume characteristics
    - Multiplicative combination of momentum acceleration and volume confirmation
    - Adaptive weights that respond to market regime changes
    - Positive values indicate strong momentum with volume confirmation across timeframes
    - Negative values suggest momentum breakdown with volume divergence patterns
    """
    
    # Hierarchical momentum components with percentile ranking
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Percentile ranking of momentum components (5-day window)
    intraday_rank = intraday_momentum.rolling(window=5).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 5 else np.nan)
    overnight_rank = overnight_momentum.rolling(window=5).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 5 else np.nan)
    weekly_rank = weekly_momentum.rolling(window=5).apply(lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) == 5 else np.nan)
    
    # Momentum acceleration hierarchy
    short_term_accel = (intraday_rank + overnight_rank) * np.sign(intraday_rank * overnight_rank)
    mid_term_accel = (overnight_rank + weekly_rank) * np.sign(overnight_rank * weekly_rank)
    combined_accel = short_term_accel * mid_term_accel * np.sign(short_term_accel * mid_term_accel)
    
    # Volume divergence detection
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_pressure = df['volume'] / (volume_5d_avg + 1e-7)
    volume_momentum_divergence = volume_pressure * np.sign(intraday_momentum + overnight_momentum + weekly_momentum)
    
    # Multi-regime classification
    daily_range = df['high'] - df['low']
    vol_5d_std = daily_range.rolling(window=5).std()
    vol_20d_median = vol_5d_std.rolling(window=20).median()
    vol_regime_ratio = vol_5d_std / (vol_20d_median + 1e-7)
    
    volume_regime_ratio = df['volume'] / (df['volume'].rolling(window=20).median() + 1e-7)
    
    # Combined regime classification
    regime_score = vol_regime_ratio * volume_regime_ratio
    market_regime = np.where(regime_score > 2.0, 'high_vol_high_vol',
                           np.where(regime_score > 1.2, 'medium',
                                  np.where(regime_score < 0.6, 'low_vol_low_vol', 'normal')))
    
    # Dynamic regime weights
    intraday_weight = np.where(market_regime == 'high_vol_high_vol', 0.4,
                              np.where(market_regime == 'low_vol_low_vol', 0.2, 0.3))
    
    overnight_weight = np.where(market_regime == 'high_vol_high_vol', 0.3,
                               np.where(market_regime == 'low_vol_low_vol', 0.3, 0.25))
    
    weekly_weight = np.where(market_regime == 'high_vol_high_vol', 0.2,
                            np.where(market_regime == 'low_vol_low_vol', 0.4, 0.3))
    
    accel_weight = np.where(market_regime == 'high_vol_high_vol', 0.1,
                           np.where(market_regime == 'low_vol_low_vol', 0.1, 0.15))
    
    # Multiplicative combination with volume confirmation
    momentum_base = (
        intraday_weight * intraday_rank +
        overnight_weight * overnight_rank +
        weekly_weight * weekly_rank +
        accel_weight * combined_accel
    )
    
    volume_confirmation = volume_momentum_divergence * np.sign(momentum_base)
    
    # Final alpha factor with hierarchical structure
    alpha_factor = momentum_base * volume_confirmation * regime_score
    
    return alpha_factor
