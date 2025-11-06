import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Hierarchical momentum acceleration with volume divergence and dynamic percentile blending.
    
    Interpretation:
    - Multi-timeframe momentum hierarchy (intraday, overnight, weekly) with acceleration signals
    - Volume divergence detection identifies momentum-validated vs momentum-contradicted periods
    - Dynamic percentile ranking creates robust cross-sectional signals across market conditions
    - Multiplicative combination of momentum acceleration and volume confirmation enhances signal strength
    - Hierarchical structure prioritizes shorter-term signals when confirmed by longer-term trends
    - Positive values indicate strong momentum acceleration with volume confirmation across timeframes
    - Negative values suggest momentum deterioration with volume divergence patterns
    """
    
    # Hierarchical momentum components with different timeframes
    intraday_momentum = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_momentum = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    weekly_momentum = (df['close'] - df['close'].shift(5)) / (df['high'].rolling(5).max() - df['low'].rolling(5).min() + 1e-7)
    
    # Momentum acceleration hierarchy
    intraday_accel = intraday_momentum - intraday_momentum.shift(1)
    overnight_accel = overnight_momentum - overnight_momentum.shift(1)
    weekly_accel = weekly_momentum - weekly_momentum.shift(3)
    
    # Volume divergence detection
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_divergence = (volume_ma_5 / (volume_ma_20 + 1e-7) - 1) * np.sign(intraday_momentum + overnight_momentum)
    
    # Multiplicative combination of momentum and volume signals
    momentum_volume_product = (intraday_momentum * overnight_momentum * weekly_momentum) * volume_divergence
    
    # Hierarchical timeframe blending with acceleration emphasis
    ultra_short_blend = intraday_momentum * intraday_accel * np.sign(intraday_momentum)
    short_term_blend = overnight_momentum * overnight_accel * np.sign(overnight_momentum)
    medium_term_blend = weekly_momentum * weekly_accel * np.sign(weekly_momentum)
    
    # Dynamic percentile ranking for robustness
    ultra_short_rank = ultra_short_blend.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)).astype(float))
    short_term_rank = short_term_blend.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)).astype(float))
    medium_term_rank = medium_term_blend.rolling(window=20).apply(lambda x: (x.iloc[-1] > x.quantile(0.6)).astype(float))
    
    # Hierarchical weighting based on timeframe confirmation
    hierarchical_weight = (
        ultra_short_rank * 0.4 + 
        short_term_rank * 0.3 + 
        medium_term_rank * 0.3
    )
    
    # Final alpha factor with hierarchical structure and multiplicative enhancement
    alpha_factor = (
        hierarchical_weight * 
        (ultra_short_blend + short_term_blend + medium_term_blend) * 
        momentum_volume_product * 
        np.sign(ultra_short_blend + short_term_blend + medium_term_blend)
    )
    
    return alpha_factor
