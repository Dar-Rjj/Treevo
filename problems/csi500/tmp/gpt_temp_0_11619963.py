import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multi-horizon momentum, volatility-normalized range,
    price-volume relationships, regime-aware scaling, and decay-adjusted signals.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Price Momentum with Volume Adjustment
    # Multi-horizon returns
    returns_short = data['close'].pct_change(3)  # 3-day return
    returns_medium = data['close'].pct_change(15)  # 15-day return  
    returns_long = data['close'].pct_change(40)  # 40-day return
    
    # Volume weighting with decay
    volume_weights = data['volume'].rolling(window=10).apply(
        lambda x: np.average(x, weights=np.exp(-np.arange(len(x))/3))
    )
    
    # Volume-adjusted momentum
    vol_adj_momentum = (returns_short * 0.4 + returns_medium * 0.35 + returns_long * 0.25) * volume_weights
    
    # 2. Volatility-Normalized Range Factors
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Volatility scaling
    vol_scale = data['close'].rolling(window=20).std()
    atr = true_range.rolling(window=14).mean()
    
    # Normalized range factor
    norm_range = (data['high'] - data['low']) / (vol_scale + atr)
    
    # 3. Price-Volume Relationship Signals
    # Volume-Price Trend
    vpt = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * data['volume']
    cum_vpt = vpt.rolling(window=10).sum()
    
    # Breakout confirmation
    high_breakout = (data['close'] > data['high'].shift(1)) & (data['volume'] > data['volume'].rolling(window=20).mean())
    low_pullback = (data['close'] < data['low'].shift(1)) & (data['volume'] < data['volume'].rolling(window=20).mean())
    breakout_signal = high_breakout.astype(int) - low_pullback.astype(int)
    
    # 4. Regime-Aware Factor Scaling
    # Market condition detection
    volatility_regime = data['close'].rolling(window=30).std() / data['close'].rolling(window=30).std().rolling(window=60).mean()
    trend_regime = data['close'].rolling(window=20).mean() / data['close'].rolling(window=60).mean() - 1
    
    # Adaptive weighting
    high_vol_weight = 1 / (1 + np.exp(-5 * (volatility_regime - 1.2)))
    low_vol_enhance = 1 + 0.5 * np.exp(-2 * abs(trend_regime))
    
    # 5. Decay-Adjusted Multi-Horizon Signals
    # Exponential weighting for recent emphasis
    def exponential_weighted_mean(series, halflife=5):
        weights = np.exp(-np.log(2) / halflife * np.arange(len(series))[::-1])
        return np.average(series, weights=weights)
    
    # Apply decay to momentum signals
    decay_momentum = vol_adj_momentum.rolling(window=10).apply(
        lambda x: exponential_weighted_mean(x, halflife=3)
    )
    
    # Combine all components with regime-aware scaling
    alpha_factor = (
        decay_momentum * 0.3 +
        norm_range * 0.25 * low_vol_enhance +
        cum_vpt * 0.2 +
        breakout_signal * 0.15 * high_vol_weight +
        trend_regime * 0.1
    )
    
    # Final smoothing and normalization
    alpha_smoothed = alpha_factor.rolling(window=5).mean()
    alpha_normalized = (alpha_smoothed - alpha_smoothed.rolling(window=60).mean()) / alpha_smoothed.rolling(window=60).std()
    
    return alpha_normalized
