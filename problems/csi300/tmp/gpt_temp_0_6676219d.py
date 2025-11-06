import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic regime adaptation.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with percentile normalization
    - Volume divergence detection across different time horizons for confirmation
    - Dynamic regime classification based on volatility and volume characteristics
    - Multiplicative combinations enhance signal strength and interpretability
    - Regime-adaptive weights optimize signal extraction across market conditions
    - Positive values indicate strong momentum with volume confirmation across multiple timeframes
    - Negative values suggest weakening momentum with volume divergence patterns
    """
    
    # Hierarchical momentum components with percentile normalization
    intraday_mom = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_mom = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_mom = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Percentile rank normalization for robustness
    intraday_rank = intraday_mom.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5)
    )
    overnight_rank = overnight_mom.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5)
    )
    weekly_rank = weekly_mom.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5)
    )
    
    # Volume divergence detection across multiple timeframes
    volume_5d_avg = df['volume'].rolling(window=5).mean()
    volume_20d_avg = df['volume'].rolling(window=20).mean()
    volume_divergence = (df['volume'] / (volume_5d_avg + 1e-7)) - (volume_5d_avg / (volume_20d_avg + 1e-7))
    
    # Multiplicative momentum combinations
    short_term_combo = intraday_rank * overnight_rank * np.sign(intraday_mom * overnight_mom)
    mid_term_combo = overnight_rank * weekly_rank * np.sign(overnight_mom * weekly_mom)
    hierarchical_combo = short_term_combo * mid_term_combo * np.sign(short_term_combo * mid_term_combo)
    
    # Dynamic regime classification
    daily_range = df['high'] - df['low']
    vol_regime = (daily_range.rolling(window=5).std() / 
                 daily_range.rolling(window=20).std().replace(0, 1e-7))
    
    volume_regime = (df['volume'].rolling(window=5).mean() / 
                    df['volume'].rolling(window=20).mean().replace(0, 1e-7))
    
    # Regime classification
    high_vol_regime = (vol_regime > 1.3).astype(int)
    low_vol_regime = (vol_regime < 0.8).astype(int)
    high_volume_regime = (volume_regime > 1.2).astype(int)
    low_volume_regime = (volume_regime < 0.9).astype(int)
    
    # Regime-adaptive weights
    intraday_weight = np.where(high_vol_regime, 0.4, 
                              np.where(low_vol_regime, 0.2, 0.3))
    overnight_weight = np.where(high_vol_regime, 0.3,
                               np.where(low_vol_regime, 0.3, 0.25))
    weekly_weight = np.where(high_vol_regime, 0.2,
                            np.where(low_vol_regime, 0.4, 0.3))
    combo_weight = np.where(high_vol_regime, 0.1,
                           np.where(low_vol_regime, 0.1, 0.15))
    
    # Volume regime adjustments
    volume_multiplier = np.where(high_volume_regime, 1.3,
                                np.where(low_volume_regime, 0.7, 1.0))
    
    # Final alpha factor with hierarchical structure
    alpha_factor = (
        intraday_weight * intraday_rank * volume_multiplier +
        overnight_weight * overnight_rank * volume_multiplier +
        weekly_weight * weekly_rank * volume_multiplier +
        combo_weight * hierarchical_combo * np.abs(volume_divergence) *
        np.sign(volume_divergence * hierarchical_combo)
    )
    
    return alpha_factor
