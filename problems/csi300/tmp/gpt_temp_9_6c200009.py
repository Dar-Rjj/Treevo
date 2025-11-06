import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum hierarchy with volume-pressure confirmation and clear regime adaptation.
    
    Interpretation:
    - Three-tier momentum hierarchy (intraday, overnight, weekly) using percentile ranks
    - Volume-pressure signals detect abnormal volume conditions relative to recent history
    - Clear regime classification based on volatility and volume characteristics
    - Multiplicative combinations enhance signal robustness across market environments
    - Positive values indicate aligned momentum across timeframes with volume confirmation
    - Negative values suggest momentum divergence or volume-pressure contradictions
    """
    
    # Multi-timeframe momentum components
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Percentile rank normalization for momentum components
    intraday_rank = intraday_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5), raw=False
    )
    overnight_rank = overnight_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5), raw=False
    )
    weekly_rank = weekly_momentum.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5), raw=False
    )
    
    # Multiplicative volume-pressure signals
    volume_pressure_3d = df['volume'] / (df['volume'].rolling(window=3).mean() + 1e-7)
    volume_pressure_8d = df['volume'] / (df['volume'].rolling(window=8).mean() + 1e-7)
    volume_pressure_15d = df['volume'] / (df['volume'].rolling(window=15).mean() + 1e-7)
    
    # Combined volume-pressure factor using multiplicative combination
    volume_pressure = volume_pressure_3d * volume_pressure_8d * volume_pressure_15d
    
    # Clear regime classification logic
    daily_range = df['high'] - df['low']
    vol_regime = daily_range.rolling(window=5).std() / (daily_range.rolling(window=20).std() + 1e-7)
    volume_regime = df['volume'].rolling(window=5).std() / (df['volume'].rolling(window=20).std() + 1e-7)
    
    # Simple regime classification
    volatility_state = np.where(vol_regime > 1.2, 'high', np.where(vol_regime < 0.8, 'low', 'normal'))
    volume_state = np.where(volume_regime > 1.2, 'high', np.where(volume_regime < 0.8, 'low', 'normal'))
    
    # Combined regime classification
    market_regime = np.where(
        (volatility_state == 'high') & (volume_state == 'high'), 'high_vol_high_vol',
        np.where(
            (volatility_state == 'low') & (volume_state == 'low'), 'low_vol_low_vol',
            np.where(
                volatility_state == 'high', 'high_vol',
                np.where(
                    volatility_state == 'low', 'low_vol',
                    'normal'
                )
            )
        )
    )
    
    # Momentum alignment factor using multiplicative combinations
    momentum_alignment = (
        intraday_rank * overnight_rank * np.sign(intraday_momentum * overnight_momentum) +
        intraday_rank * weekly_rank * np.sign(intraday_momentum * weekly_momentum) +
        overnight_rank * weekly_rank * np.sign(overnight_momentum * weekly_momentum)
    )
    
    # Regime-adaptive momentum weights
    intraday_weight = np.where(
        market_regime == 'high_vol_high_vol', 0.4,
        np.where(
            market_regime == 'low_vol_low_vol', 0.2,
            np.where(
                market_regime == 'high_vol', 0.35,
                np.where(
                    market_regime == 'low_vol', 0.25,
                    0.3
                )
            )
        )
    )
    
    overnight_weight = np.where(
        market_regime == 'high_vol_high_vol', 0.3,
        np.where(
            market_regime == 'low_vol_low_vol', 0.4,
            np.where(
                market_regime == 'high_vol', 0.25,
                np.where(
                    market_regime == 'low_vol', 0.35,
                    0.3
                )
            )
        )
    )
    
    weekly_weight = np.where(
        market_regime == 'high_vol_high_vol', 0.3,
        np.where(
            market_regime == 'low_vol_low_vol', 0.4,
            np.where(
                market_regime == 'high_vol', 0.4,
                np.where(
                    market_regime == 'low_vol', 0.4,
                    0.4
                )
            )
        )
    )
    
    # Volume-pressure confirmation signal
    volume_confirmation = np.where(
        (intraday_momentum > 0) & (volume_pressure > 1.0), 1.0,
        np.where(
            (intraday_momentum < 0) & (volume_pressure < 1.0), 1.0,
            np.where(
                (intraday_momentum > 0) & (volume_pressure < 1.0), -1.0,
                np.where(
                    (intraday_momentum < 0) & (volume_pressure > 1.0), -1.0,
                    0.0
                )
            )
        )
    )
    
    # Final alpha factor with clear hierarchical structure
    alpha_factor = (
        intraday_weight * intraday_rank +
        overnight_weight * overnight_rank +
        weekly_weight * weekly_rank +
        momentum_alignment * 0.2 +
        volume_confirmation * 0.3
    )
    
    return alpha_factor
