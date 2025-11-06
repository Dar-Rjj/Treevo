import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with percentile normalization
    - Volume divergence detection across different time horizons for signal confirmation
    - Dynamic regime classification based on volatility and volume characteristics
    - Multiplicative combination of momentum and volume components for enhanced signal strength
    - Regime-adaptive weights that respond to changing market conditions
    - Positive values indicate strong momentum with volume confirmation across multiple timeframes
    - Negative values suggest momentum breakdown with volume divergence patterns
    """
    
    # Hierarchical momentum components with percentile normalization
    intraday_return = (df['close'] - df['open']) / df['open']
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Percentile rank normalization for robustness
    intraday_rank = intraday_return.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    overnight_rank = overnight_return.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    weekly_rank = weekly_momentum.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    
    # Volume divergence detection across timeframes
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    short_volume_divergence = df['volume'] / (volume_5d_avg + 1e-7)
    long_volume_divergence = df['volume'] / (volume_20d_avg + 1e-7)
    
    # Volume divergence percentile ranks
    short_vol_rank = short_volume_divergence.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    long_vol_rank = long_volume_divergence.rolling(window=20).apply(lambda x: (x.iloc[-1] > x).mean())
    
    # Dynamic regime classification
    daily_range = (df['high'] - df['low']) / df['close']
    vol_regime = daily_range.rolling(window=5).std()
    vol_regime_median = vol_regime.rolling(window=20).median()
    volatility_state = vol_regime / (vol_regime_median + 1e-7)
    
    # Multiplicative momentum-volume combinations
    intraday_momentum_volume = intraday_rank * short_vol_rank * np.sign(intraday_return)
    overnight_momentum_volume = overnight_rank * short_vol_rank * np.sign(overnight_return)
    weekly_momentum_volume = weekly_rank * long_vol_rank * np.sign(weekly_momentum)
    
    # Momentum acceleration hierarchy
    ultra_short_accel = (intraday_momentum_volume + overnight_momentum_volume) * np.sign(intraday_momentum_volume * overnight_momentum_volume)
    medium_term_accel = (overnight_momentum_volume + weekly_momentum_volume) * np.sign(overnight_momentum_volume * weekly_momentum_volume)
    hierarchical_accel = (ultra_short_accel + medium_term_accel) * np.sign(ultra_short_accel * medium_term_accel)
    
    # Volume divergence confirmation
    volume_convergence = (short_vol_rank + long_vol_rank) * np.sign(short_vol_rank * long_vol_rank)
    volume_momentum_alignment = volume_convergence * (intraday_rank + overnight_rank + weekly_rank)
    
    # Dynamic regime weights
    high_vol_weight = np.where(volatility_state > 1.5, 0.6, 0.3)
    medium_vol_weight = np.where((volatility_state >= 0.8) & (volatility_state <= 1.5), 0.8, 0.4)
    low_vol_weight = np.where(volatility_state < 0.8, 0.5, 0.2)
    
    # Regime-adaptive factor combination
    alpha_factor = (
        high_vol_weight * hierarchical_accel +
        medium_vol_weight * volume_momentum_alignment +
        low_vol_weight * (intraday_momentum_volume + weekly_momentum_volume) +
        (intraday_momentum_volume * overnight_momentum_volume * weekly_momentum_volume) * 0.1
    )
    
    return alpha_factor
