import pandas as pd
import pandas as pd

def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive Multi-Horizon Price-Volume Divergence with Liquidity Amplification
    
    This factor combines price-volume divergence across two complementary time horizons
    (3-day and 8-day) multiplicatively, adapts to volatility regimes through dynamic
    smoothing, and amplifies signals during high liquidity conditions.
    
    Key features:
    - Dual-horizon price-volume divergence (3-day and 8-day) for robustness
    - Liquidity amplification using dollar volume momentum
    - Volatility regime detection via rolling high-low range percentiles
    - Regime-adaptive exponential smoothing with dynamic spans
    - Pure multiplicative combination for non-linear signal enhancement
    
    The factor identifies stocks with consistent divergence patterns across timeframes,
    amplified by liquidity conditions, and adaptively smoothed based on market volatility.
    """
    # Dual-horizon price momentum (3-day and 8-day)
    price_momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    price_momentum_8d = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    # Dual-horizon volume momentum with rolling mean
    volume_roll_3d = df['volume'].rolling(window=3).mean()
    volume_roll_8d = df['volume'].rolling(window=8).mean()
    
    volume_momentum_3d = (df['volume'] - volume_roll_3d) / (volume_roll_3d + 1e-7)
    volume_momentum_8d = (df['volume'] - volume_roll_8d) / (volume_roll_8d + 1e-7)
    
    # Dual-horizon price-volume divergence
    divergence_3d = price_momentum_3d * (1 - volume_momentum_3d)
    divergence_8d = price_momentum_8d * (1 - volume_momentum_8d)
    
    # Multiplicative combination of dual horizons
    multi_horizon_divergence = divergence_3d * divergence_8d
    
    # Liquidity amplification using dollar volume momentum
    dollar_volume = df['close'] * df['volume']
    dollar_volume_roll_5d = dollar_volume.rolling(window=5).mean()
    liquidity_amplifier = dollar_volume / (dollar_volume_roll_5d + 1e-7)
    
    # Liquidity-amplified divergence
    amplified_divergence = multi_horizon_divergence * liquidity_amplifier
    
    # Volatility regime detection using high-low range
    high_low_range = (df['high'] - df['low']) / df['close']
    
    # Rolling percentile-based regime classification
    volatility_percentile = high_low_range.rolling(window=15, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).mean() if len(x) > 1 else 0.5
    )
    
    # Define volatility regimes
    high_vol_regime = volatility_percentile > 0.7
    medium_vol_regime = (volatility_percentile >= 0.4) & (volatility_percentile <= 0.7)
    low_vol_regime = volatility_percentile < 0.4
    
    # Regime-adaptive exponential smoothing
    # High volatility: more smoothing (longer span)
    # Low volatility: more responsive (shorter span)
    alpha_factor_high_vol = amplified_divergence[high_vol_regime].ewm(span=6).mean()
    alpha_factor_medium_vol = amplified_divergence[medium_vol_regime].ewm(span=4).mean()
    alpha_factor_low_vol = amplified_divergence[low_vol_regime].ewm(span=2).mean()
    
    # Combine regime-adaptive factors
    alpha_factor = pd.concat([
        alpha_factor_high_vol,
        alpha_factor_medium_vol,
        alpha_factor_low_vol
    ]).sort_index()
    
    return alpha_factor
