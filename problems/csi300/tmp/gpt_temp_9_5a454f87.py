import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Price Acceleration with Regime Switching
    # Compute second derivative of price (momentum of momentum)
    momentum = data['close'].pct_change(periods=5)
    acceleration = momentum.diff(periods=3)
    
    # Identify regime changes using price curvature
    price_curvature = data['close'].pct_change(periods=2) - 2 * data['close'].pct_change(periods=1) + data['close'].pct_change(periods=3)
    regime_change = (price_curvature.rolling(window=10).std() > price_curvature.rolling(window=30).std()).astype(int)
    
    # Generate regime-adaptive acceleration signals
    regime_adaptive_accel = acceleration * (1 + 0.5 * regime_change)
    
    # Bid-Ask Imbalance Proxy
    # Estimate bid-ask spread using high-low range
    relative_spread = (data['high'] - data['low']) / data['close']
    
    # Calculate intraday price pressure direction
    intraday_pressure = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Generate imbalance momentum signals
    imbalance_momentum = intraday_pressure.rolling(window=5).mean() * (1 - relative_spread.rolling(window=5).mean())
    
    # Volume-Volatility Convexity
    # Compute volatility sensitivity to volume changes
    daily_volatility = (data['high'] - data['low']) / data['close']
    volume_change = data['volume'].pct_change(periods=1)
    vol_vol_sensitivity = daily_volatility.diff() / volume_change.replace(0, np.nan)
    
    # Identify convex/concave volume-volatility relationships
    vol_vol_convexity = vol_vol_sensitivity.diff(periods=3)
    
    # Generate convexity-based reversal signals
    convexity_signal = -vol_vol_convexity.rolling(window=10).mean()
    
    # Overnight Gap Persistence
    # Calculate overnight return gaps (open vs previous close)
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Measure gap persistence across multiple days
    gap_persistence = overnight_gap.rolling(window=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0)
    
    # Generate gap continuation/mean reversion signals
    gap_signal = overnight_gap * gap_persistence
    
    # Price-Volume Fractal Dimension
    # Compute multi-scale price-volume correlation structure
    corr_1d = data['close'].pct_change(periods=1).rolling(window=10).corr(data['volume'].pct_change(periods=1))
    corr_3d = data['close'].pct_change(periods=3).rolling(window=10).corr(data['volume'].pct_change(periods=3))
    corr_5d = data['close'].pct_change(periods=5).rolling(window=10).corr(data['volume'].pct_change(periods=5))
    
    # Calculate fractal dimension of price-volume relationship
    fractal_dim = (corr_5d - corr_3d) / (corr_3d - corr_1d).replace(0, np.nan)
    
    # Generate complexity-based trend signals
    complexity_signal = fractal_dim.rolling(window=5).mean()
    
    # Extreme Value Compression
    # Identify compression periods using daily range percentiles
    daily_range = (data['high'] - data['low']) / data['close']
    range_percentile = daily_range.rolling(window=20).apply(lambda x: (x.iloc[-1] <= np.percentile(x, 25)) if len(x) == 20 else np.nan)
    
    # Measure duration of low-volatility compression
    compression_duration = range_percentile.rolling(window=10).sum()
    
    # Generate breakout anticipation signals
    breakout_signal = -compression_duration * daily_range
    
    # Cross-Sectional Momentum Decay
    # Calculate relative momentum decay rates
    momentum_5d = data['close'].pct_change(periods=5)
    momentum_10d = data['close'].pct_change(periods=10)
    momentum_decay = (momentum_5d - momentum_10d) / 5
    
    # Identify momentum acceleration/deceleration patterns
    momentum_accel = momentum_decay.diff(periods=3)
    
    # Generate decay-adjusted momentum signals
    decay_adjusted_momentum = momentum_5d * (1 - 0.2 * momentum_accel.rolling(window=5).mean())
    
    # Combine all signals with equal weights
    combined_factor = (
        regime_adaptive_accel.rank(pct=True) +
        imbalance_momentum.rank(pct=True) +
        convexity_signal.rank(pct=True) +
        gap_signal.rank(pct=True) +
        complexity_signal.rank(pct=True) +
        breakout_signal.rank(pct=True) +
        decay_adjusted_momentum.rank(pct=True)
    )
    
    # Normalize the final factor
    final_factor = (combined_factor - combined_factor.rolling(window=20).mean()) / combined_factor.rolling(window=20).std()
    
    return final_factor
