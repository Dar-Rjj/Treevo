import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Enhanced alpha factor blending asymmetric decay with volatility-normalized returns,
    volume-microstructure alignment, and regime-adaptive scaling for robust predictive signals.
    """
    
    # Core price returns for momentum signals
    returns = df['close'].pct_change()
    
    # Volatility regime detection using asymmetric rolling windows
    short_vol = returns.rolling(window=3, min_periods=1).std() + 1e-7
    long_vol = returns.rolling(window=10, min_periods=1).std() + 1e-7
    vol_regime = (short_vol / long_vol).clip(upper=2.0, lower=0.5)
    
    # Asymmetric volatility-normalized returns with regime adaptation
    normalized_returns = returns / (short_vol * vol_regime)
    
    # Directional momentum with asymmetric decay profiles
    pos_returns = normalized_returns.where(normalized_returns > 0, 0)
    neg_returns = normalized_returns.where(normalized_returns < 0, 0)
    
    # Fast decay for positive momentum, slow decay for negative momentum
    pos_momentum = pos_returns.ewm(span=2, adjust=False).mean()
    neg_momentum = neg_returns.ewm(span=6, adjust=False).mean()
    directional_momentum = pos_momentum - neg_momentum
    
    # Volume microstructure alignment with trade intensity
    volume_returns = df['volume'].pct_change()
    
    # Asymmetric volume response - faster reaction to increasing volume
    pos_volume = volume_returns.where(volume_returns > 0, 0).ewm(span=2, adjust=False).mean()
    neg_volume = volume_returns.where(volume_returns < 0, 0).ewm(span=4, adjust=False).mean()
    volume_pressure = pos_volume - neg_volume
    
    # Volume-confirmed momentum
    volume_aligned_momentum = directional_momentum * volume_pressure
    
    # Price range efficiency with gap analysis
    daily_range = df['high'] - df['low'] + 1e-7
    range_efficiency = (df['close'] - df['open']) / daily_range
    
    # Gap persistence with directional consistency
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    gap_strength = overnight_gap.rolling(window=3, min_periods=1).apply(
        lambda x: np.mean(x) if len([v for v in x if np.sign(v) == np.sign(np.mean(x))]) >= 2 else 0
    )
    
    range_momentum = range_efficiency * gap_strength
    
    # Microstructure pressure signals
    prev_close = df['close'].shift(1)
    high_rejection = ((df['high'] - prev_close) / daily_range).ewm(span=2, adjust=False).mean()
    low_support = ((prev_close - df['low']) / daily_range).ewm(span=3, adjust=False).mean()
    
    # Trade efficiency with amount-volume divergence
    avg_trade_size = df['amount'] / (df['volume'] + 1e-7)
    trade_efficiency = avg_trade_size.pct_change(periods=2)
    
    microstructure_signal = (high_rejection - low_support) * trade_efficiency
    
    # Regime-adaptive scaling based on recent performance
    recent_performance = returns.rolling(window=5, min_periods=1).mean()
    regime_weight = np.where(recent_performance > 0, 1.2, 0.8)
    
    # Composite factor with regime-adaptive weights
    factor = (
        0.40 * volume_aligned_momentum * regime_weight +
        0.35 * microstructure_signal +
        0.25 * range_momentum
    )
    
    return factor
