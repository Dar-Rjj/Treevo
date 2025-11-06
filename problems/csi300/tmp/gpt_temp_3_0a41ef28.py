import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using second-order derivative features, multi-timeframe alignment, 
    and regime-adaptive dynamic features.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Price and volume calculations
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # 1. Second-Order Derivative Features
    # Price acceleration (2nd derivative of close price)
    price_acceleration = close.diff().diff()
    
    # Volume acceleration (2nd derivative of volume)
    volume_acceleration = volume.diff().diff()
    
    # Price Acceleration Regime Indicator
    regime_signal = price_acceleration * volume_acceleration
    
    # Volatility Acceleration Divergence
    daily_range = high - low
    range_acceleration = daily_range.diff().diff()
    volatility_divergence = range_acceleration - price_acceleration
    
    # Opening Gap Momentum Acceleration
    gap_series = (open_price - close.shift(1)) / close.shift(1)
    gap_acceleration = gap_series.diff().diff()
    intraday_momentum = (close - open_price) / open_price
    gap_momentum = gap_acceleration * intraday_momentum.rolling(window=3).mean()
    
    # 2. Multi-Timeframe Alignment Features
    # Triple-Timeframe Acceleration Convergence
    price_accel_5d = close.pct_change(periods=5).diff().diff()
    price_accel_10d = close.pct_change(periods=10).diff().diff()
    price_accel_20d = close.pct_change(periods=20).diff().diff()
    
    # Convergence score (standard deviation of accelerations)
    accel_convergence = pd.concat([price_accel_5d, price_accel_10d, price_accel_20d], axis=1).std(axis=1)
    
    # Volume acceleration alignment
    vol_accel_5d = volume.pct_change(periods=5).diff().diff()
    vol_accel_10d = volume.pct_change(periods=10).diff().diff()
    vol_accel_20d = volume.pct_change(periods=20).diff().diff()
    volume_alignment = pd.concat([vol_accel_5d, vol_accel_10d, vol_accel_20d], axis=1).std(axis=1)
    
    triple_timeframe_signal = accel_convergence * volume_alignment
    
    # Volatility Regime Transition Acceleration
    short_term_vol = close.rolling(window=5).std().pct_change()
    long_term_vol = close.rolling(window=20).std().pct_change()
    short_vol_accel = short_term_vol.diff().diff()
    long_vol_accel = long_term_vol.diff().diff()
    vol_regime_transition = short_vol_accel - long_vol_accel
    
    # Volume-Price Acceleration Alignment
    price_accel_multi = pd.concat([price_accel_5d, price_accel_10d, price_accel_20d], axis=1)
    vol_accel_multi = pd.concat([vol_accel_5d, vol_accel_10d, vol_accel_20d], axis=1)
    
    # Correlation between price and volume acceleration across timeframes
    alignment_scores = []
    for i in range(len(price_accel_multi.columns)):
        if i >= len(vol_accel_multi.columns):
            break
        corr = price_accel_multi.iloc[:, i].rolling(window=10).corr(vol_accel_multi.iloc[:, i])
        alignment_scores.append(corr)
    
    volume_price_alignment = pd.concat(alignment_scores, axis=1).mean(axis=1)
    
    # 3. Regime-Adaptive Dynamic Features
    # Acceleration-Based Mean Reversion
    acceleration_extreme = price_acceleration.rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    mean_reversion_potential = -acceleration_extreme * volume_acceleration.rolling(window=5).mean()
    
    # Dynamic Breakout Confirmation
    price_accel_magnitude = price_acceleration.abs()
    volume_accel_alignment = (price_acceleration * volume_acceleration).rolling(window=5).mean()
    breakout_signal = price_accel_magnitude * volume_accel_alignment
    
    # Regime-Adaptive Momentum Enhancement
    momentum_regime = price_acceleration.rolling(window=10).apply(
        lambda x: 1 if x.mean() > 0 else (-1 if x.mean() < 0 else 0)
    )
    regime_momentum = momentum_regime * close.pct_change(periods=5)
    multi_timeframe_alignment = pd.concat([price_accel_5d, price_accel_10d], axis=1).mean(axis=1)
    adaptive_momentum = regime_momentum * multi_timeframe_alignment
    
    # Combine all features with weights
    result = (
        0.15 * regime_signal.rank(pct=True) +
        0.12 * volatility_divergence.rank(pct=True) +
        0.10 * gap_momentum.rank(pct=True) +
        0.13 * triple_timeframe_signal.rank(pct=True) +
        0.11 * vol_regime_transition.rank(pct=True) +
        0.12 * volume_price_alignment.rank(pct=True) +
        0.14 * mean_reversion_potential.rank(pct=True) +
        0.08 * breakout_signal.rank(pct=True) +
        0.05 * adaptive_momentum.rank(pct=True)
    )
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
