import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence using percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum acceleration (intraday, overnight, multi-day) with hierarchical weighting
    - Volume divergence detection across multiple lookback periods for regime confirmation
    - Percentile-based regime classification for smooth transitions between market states
    - Multiplicative combinations enhance signal robustness while maintaining interpretability
    - Dynamic weighting adapts to momentum-velocity relationships across timeframes
    - Positive values indicate accelerating bullish momentum with volume confirmation
    - Negative values suggest deteriorating momentum with volume distribution patterns
    """
    
    # Multi-timeframe momentum components
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    daily_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    three_day_return = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    daily_accel = daily_return - daily_return.shift(1)
    multi_day_accel = three_day_return - three_day_return.shift(3)
    
    # Volume divergence detection
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_10d_avg = df['volume'].rolling(window=10).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    
    short_term_volume_div = (df['volume'] - volume_5d_avg) / (volume_5d_avg + 1e-7)
    medium_term_volume_div = (df['volume'] - volume_10d_avg) / (volume_10d_avg + 1e-7)
    long_term_volume_div = (df['volume'] - volume_20d_avg) / (volume_20d_avg + 1e-7)
    
    # Percentile-based regime classification
    momentum_percentile = intraday_return.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_percentile = short_term_volume_div.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volatility_percentile = ((df['high'] - df['low']) / df['close']).rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Smooth regime transitions using percentile combinations
    momentum_regime = np.where(momentum_percentile > 0.7, 1.8,
                              np.where(momentum_percentile < 0.3, 0.4, 1.2))
    
    volume_regime = np.where(volume_percentile > 0.7, 1.6,
                            np.where(volume_percentile < 0.3, 0.6, 1.1))
    
    volatility_regime = np.where(volatility_percentile > 0.7, 0.7,
                                np.where(volatility_percentile < 0.3, 1.4, 1.0))
    
    # Multi-timeframe momentum convergence
    ultra_short_convergence = intraday_accel * overnight_accel * np.sign(intraday_accel + overnight_accel)
    short_term_convergence = daily_accel * multi_day_accel * np.sign(daily_accel + multi_day_accel)
    cross_timeframe_convergence = ultra_short_convergence * short_term_convergence * np.sign(ultra_short_convergence + short_term_convergence)
    
    # Volume divergence confirmation
    volume_momentum_alignment = (
        short_term_volume_div * intraday_accel * np.sign(short_term_volume_div * intraday_accel) +
        medium_term_volume_div * daily_accel * np.sign(medium_term_volume_div * daily_accel) +
        long_term_volume_div * multi_day_accel * np.sign(long_term_volume_div * multi_day_accel)
    )
    
    # Dynamic regime-adaptive weights
    intraday_weight = momentum_regime * volatility_regime * 0.35
    overnight_weight = momentum_regime * volume_regime * 0.25
    daily_weight = volume_regime * volatility_regime * 0.20
    acceleration_weight = momentum_regime * volume_regime * volatility_regime * 0.20
    
    # Multiplicative alpha factor combination
    alpha_factor = (
        intraday_weight * intraday_accel *
        overnight_weight * overnight_accel *
        daily_weight * daily_accel *
        acceleration_weight * cross_timeframe_convergence *
        volume_momentum_alignment
    )
    
    return alpha_factor
