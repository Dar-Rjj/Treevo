import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Alpha Factor
    Adapts momentum and mean reversion signals based on volatility regimes
    """
    
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    # Historical volatility calculation
    hist_vol = returns.rolling(window=20, min_periods=10).std()
    vol_percentile = hist_vol.rolling(window=100, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 50 else np.nan, 
        raw=False
    )
    
    # Regime classification
    high_vol_regime = (vol_percentile > 0.6).astype(int)
    low_vol_regime = (vol_percentile < 0.4).astype(int)
    regime_strength = np.abs(vol_percentile - 0.5) * 2  # 0 to 1 scale
    
    # Multi-Timeframe Momentum Alignment
    # Short-term momentum (3-day)
    short_momentum = df['close'].pct_change(3)
    vwap_3d = (df['close'] * df['volume']).rolling(3).sum() / df['volume'].rolling(3).sum()
    short_vw_momentum = (df['close'] - vwap_3d) / vwap_3d
    
    # Medium-term momentum (10-day)
    medium_momentum = df['close'].pct_change(10)
    volume_trend = df['volume'].rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else np.nan,
        raw=False
    )
    
    # Long-term momentum (20-day)
    long_momentum = df['close'].pct_change(20)
    directional_persistence = returns.rolling(20).apply(
        lambda x: np.sum(np.sign(x) == np.sign(np.nanmean(x))) if len(x) >= 10 else np.nan,
        raw=False
    ) / 20
    
    # High Volatility Regime Factor
    # Mean reversion components
    recent_overreaction = (
        (df['high'] - df['low']).rolling(5).mean() / df['close'].rolling(20).std() -
        (df['high'] - df['low']).rolling(20).mean() / df['close'].rolling(20).std()
    )
    
    volume_decline = df['volume'].pct_change(5) < -0.1
    extreme_ranges = (df['high'] - df['low']) / df['close'] > 0.03
    
    volatility_contraction = (
        (df['high'] - df['low']).rolling(5).std() / 
        (df['high'] - df['low']).rolling(20).std()
    )
    
    volume_normalization = df['volume'] / df['volume'].rolling(20).mean()
    
    high_vol_factor = (
        -recent_overreaction * 0.4 +
        (volume_decline & extreme_ranges).astype(float) * 0.3 +
        volatility_contraction * 0.2 -
        volume_normalization * 0.1
    )
    
    # Momentum dampening in high vol
    high_vol_momentum = (
        short_momentum * 0.6 + 
        short_vw_momentum * 0.4
    ) * 0.5  # Reduced weight
    
    # Low Volatility Regime Factor
    # Trend following components
    momentum_alignment = (
        (np.sign(short_momentum) == np.sign(medium_momentum)).astype(int) +
        (np.sign(medium_momentum) == np.sign(long_momentum)).astype(int) +
        (np.sign(short_momentum) == np.sign(long_momentum)).astype(int)
    ) / 3
    
    alignment_magnitude = (
        np.abs(short_momentum) * np.abs(medium_momentum) * np.abs(long_momentum)
    ) ** (1/3)
    
    # Breakout confirmation
    volume_expansion = df['volume'] > df['volume'].rolling(20).mean() * 1.2
    price_persistence = (df['close'] > df['close'].rolling(10).mean()).rolling(5).sum() / 5
    
    # Volatility expansion anticipation
    range_expansion = (df['high'] - df['low']).pct_change(5) > 0
    volume_buildup = df['volume'].rolling(5).mean() > df['volume'].rolling(20).mean()
    
    relative_range = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(20).mean()
    range_acceleration = (df['high'] - df['low']).diff(3)
    
    low_vol_factor = (
        momentum_alignment * alignment_magnitude * 0.4 +
        (volume_expansion & (price_persistence > 0.6)).astype(float) * 0.3 +
        (range_expansion & volume_buildup).astype(float) * 0.2 +
        (relative_range * range_acceleration).clip(-1, 1) * 0.1
    )
    
    # Low vol momentum (full weight)
    low_vol_momentum = (
        short_momentum * 0.3 +
        medium_momentum * 0.4 +
        long_momentum * 0.3
    )
    
    # Adaptive Signal Combination
    # Regime-weighted factors
    regime_weighted_factor = (
        high_vol_regime * regime_strength * (high_vol_factor + high_vol_momentum) +
        low_vol_regime * regime_strength * (low_vol_factor + low_vol_momentum)
    )
    
    # Dynamic lookback adjustment
    lookback_adjustment = np.where(
        high_vol_regime == 1,
        regime_weighted_factor.rolling(5).mean(),
        regime_weighted_factor.rolling(10).mean()
    )
    
    # Volume confirmation filter
    volume_volatility = df['volume'].rolling(20).std() / df['volume'].rolling(20).mean()
    volume_trend_confirmation = volume_trend * np.sign(lookback_adjustment)
    
    volume_weighted_factor = lookback_adjustment * (1 + volume_trend_confirmation.clip(-0.5, 0.5))
    
    # Volume divergence alerts
    price_volume_corr = df['close'].pct_change().rolling(10).corr(df['volume'].pct_change())
    volume_divergence = (price_volume_corr < -0.3).astype(float) * -0.2
    
    final_factor = volume_weighted_factor + volume_divergence
    
    # Normalize and clean
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    final_factor = (final_factor - final_factor.rolling(50).mean()) / final_factor.rolling(50).std()
    
    return final_factor
