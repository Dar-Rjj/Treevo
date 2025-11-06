import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Multi-timeframe momentum acceleration with volume divergence via percentile-based regime weights.
    
    Interpretation:
    - Triple-timeframe momentum hierarchy (intraday, overnight, multi-day) with acceleration signals
    - Volume divergence detection across different momentum regimes
    - Percentile-based regime classification for smooth transitions between market states
    - Multiplicative combinations enhance signal robustness and interpretability
    - Dynamic regime persistence ensures factor stability across market cycles
    - Volume-momentum synchronization provides confirmation of price movement validity
    - Positive values indicate strong momentum acceleration with volume confirmation
    - Negative values suggest momentum deceleration with volume divergence
    """
    
    # Core momentum components with acceleration
    intraday_return = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-7)
    overnight_return = (df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-7)
    multi_day_return = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(3).max() - df['low'].rolling(3).min() + 1e-7)
    
    # Momentum acceleration signals
    intraday_accel = intraday_return - intraday_return.shift(1)
    overnight_accel = overnight_return - overnight_return.shift(1)
    multi_day_accel = multi_day_return - multi_day_return.shift(2)
    
    # Volume divergence components
    volume_ma_5 = df['volume'].rolling(window=5).mean()
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_divergence_short = df['volume'] / (volume_ma_5 + 1e-7)
    volume_divergence_long = df['volume'] / (volume_ma_20 + 1e-7)
    
    # Price-volume synchronization
    price_volume_sync = intraday_return * np.sign(volume_divergence_short - 1)
    momentum_volume_sync = multi_day_accel * np.sign(volume_divergence_long - 1)
    
    # Percentile-based regime classification
    vol_5d = (df['high'] - df['low']).rolling(window=5).std()
    vol_regime_percentile = vol_5d.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.33)) / (x.quantile(0.67) - x.quantile(0.33) + 1e-7))
    
    volume_regime_percentile = volume_divergence_short.rolling(window=20).apply(lambda x: (x.iloc[-1] - x.quantile(0.33)) / (x.quantile(0.67) - x.quantile(0.33) + 1e-7))
    
    # Smooth regime transitions using sigmoid-like functions
    vol_regime_weight = 1 / (1 + np.exp(-3 * (vol_regime_percentile - 0.5)))
    volume_regime_weight = 1 / (1 + np.exp(-3 * (volume_regime_percentile - 0.5)))
    
    # Dynamic regime persistence
    regime_persistence = (vol_regime_weight.rolling(window=3).mean() + 
                         volume_regime_weight.rolling(window=3).mean()) / 2
    
    # Multi-timeframe momentum hierarchy with acceleration
    ultra_short_momentum = (intraday_accel + overnight_accel) * np.sign(intraday_accel * overnight_accel)
    short_term_momentum = (overnight_accel + multi_day_accel) * np.sign(overnight_accel * multi_day_accel)
    combined_momentum = (ultra_short_momentum + short_term_momentum) * np.sign(ultra_short_momentum * short_term_momentum)
    
    # Volume divergence momentum signals
    volume_confirmed_momentum = combined_momentum * volume_divergence_short
    volume_divergence_momentum = combined_momentum * (volume_divergence_short - volume_divergence_long)
    
    # Regime-adaptive weights with multiplicative combinations
    momentum_weight = 0.6 * regime_persistence
    volume_weight = 0.4 * (1 - regime_persistence)
    
    # Multiplicative factor combination for enhanced robustness
    momentum_component = (intraday_accel * overnight_accel * multi_day_accel) ** (1/3)
    volume_component = (volume_confirmed_momentum * volume_divergence_momentum) ** 0.5
    
    # Final alpha factor with regime-adaptive blending
    alpha_factor = (
        momentum_weight * momentum_component +
        volume_weight * volume_component +
        0.1 * (momentum_component * volume_component) * regime_persistence
    )
    
    return alpha_factor
