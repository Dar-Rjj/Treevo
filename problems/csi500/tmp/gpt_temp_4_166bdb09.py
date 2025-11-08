import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Horizon Momentum Decay Factor
    close = df['close']
    volume = df['volume']
    
    # Calculate multi-period returns
    r1 = close.pct_change(1)
    r3 = close.pct_change(3)
    r5 = close.pct_change(5)
    r10 = close.pct_change(10)
    
    # Apply exponential decay weighting
    w1, w3, w5, w10 = 0.4, 0.3, 0.2, 0.1
    weighted_momentum = w1 * r1 + w3 * r3 + w5 * r5 + w10 * r10
    
    # Volume momentum
    volume_momentum = volume.pct_change(5)
    price_volume_momentum = weighted_momentum * volume_momentum
    
    # Volume-Weighted Range Persistence
    high, low = df['high'], df['low']
    normalized_range = (high - low) / close
    volume_trend = volume.pct_change(5)
    range_volume = normalized_range * volume_trend
    range_persistence = range_volume.rolling(3).sum()
    
    # Intraday Momentum Efficiency
    open_price = df['open']
    intraday_strength = (close - open_price) / (high - low).replace(0, np.nan)
    short_volume_trend = volume.pct_change(3)
    medium_volume_trend = volume.pct_change(10)
    volume_acceleration = short_volume_trend - medium_volume_trend
    intraday_signal = intraday_strength * volume_acceleration
    
    # Apply momentum decay weighting for intraday signal
    weights = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    intraday_weighted = intraday_signal.rolling(5).apply(
        lambda x: np.nansum(x * weights[:len(x)]), raw=False
    )
    
    # Price-Volume Convergence Divergence
    price_momentum_short = close.pct_change(3)
    price_momentum_long = close.pct_change(10)
    price_ratio = price_momentum_short / price_momentum_long.replace(0, np.nan)
    
    volume_momentum_short = volume.pct_change(3)
    volume_momentum_long = volume.pct_change(10)
    volume_ratio = volume_momentum_short / volume_momentum_long.replace(0, np.nan)
    
    convergence_score = price_ratio * volume_ratio
    
    # Compression-Expansion Momentum
    current_range = high - low
    avg_range = current_range.rolling(5).mean()
    compression = (current_range / avg_range.replace(0, np.nan) - 1)
    
    # Breakout detection
    short_high = high.rolling(3).max()
    short_low = low.rolling(3).min()
    medium_high = high.rolling(10).max()
    medium_low = low.rolling(10).min()
    
    short_breakout = ((close > short_high.shift(1)) | (close < short_low.shift(1))).astype(float)
    medium_breakout = ((close > medium_high.shift(1)) | (close < medium_low.shift(1))).astype(float)
    breakout_strength = short_breakout + medium_breakout
    
    compression_breakout = compression * breakout_strength * volume_momentum
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'momentum': price_volume_momentum,
        'range_persistence': range_persistence,
        'intraday': intraday_weighted,
        'convergence': convergence_score,
        'compression': compression_breakout
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
