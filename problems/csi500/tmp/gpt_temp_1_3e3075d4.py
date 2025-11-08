import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Volatility-Normalized Momentum with Volume Divergence
    """
    df = data.copy()
    
    # Calculate daily returns for volatility calculation
    daily_returns = df['close'].pct_change()
    
    # Volatility-Normalized Momentum Calculation
    # Short-term momentum (3-day)
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    short_vol = daily_returns.rolling(window=10, min_periods=5).std()
    short_norm_momentum = short_momentum / short_vol.replace(0, np.nan)
    
    # Medium-term momentum (10-day)
    medium_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    medium_vol = daily_returns.rolling(window=20, min_periods=10).std()
    medium_norm_momentum = medium_momentum / medium_vol.replace(0, np.nan)
    
    # Long-term momentum (20-day)
    long_momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    long_vol = daily_returns.rolling(window=40, min_periods=20).std()
    long_norm_momentum = long_momentum / long_vol.replace(0, np.nan)
    
    # Volume Divergence Analysis
    # Price-Volume Trend Comparison
    price_trend = df['close'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    trend_divergence = price_trend - volume_trend
    
    # Volume Momentum vs Price Momentum
    volume_momentum = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    price_momentum_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    volume_price_ratio = volume_momentum / price_momentum_5d.replace(0, np.nan)
    
    # Combine volume divergence signals
    volume_divergence = 0.6 * trend_divergence + 0.4 * volume_price_ratio
    
    # Regime-Based Weighting
    current_vol = daily_returns.rolling(window=10, min_periods=5).std()
    long_term_vol = daily_returns.rolling(window=60, min_periods=30).std()
    very_long_vol = daily_returns.rolling(window=40, min_periods=20).std()
    
    # Volatility regime detection
    high_vol_regime = current_vol > long_term_vol
    low_vol_regime = current_vol < very_long_vol
    normal_regime = ~high_vol_regime & ~low_vol_regime
    
    # Initialize regime weights
    short_weight = pd.Series(0.33, index=df.index)
    medium_weight = pd.Series(0.33, index=df.index)
    long_weight = pd.Series(0.34, index=df.index)
    
    # Adjust weights based on volatility regime
    short_weight[high_vol_regime] = 0.6
    medium_weight[high_vol_regime] = 0.3
    long_weight[high_vol_regime] = 0.1
    
    short_weight[low_vol_regime] = 0.1
    medium_weight[low_vol_regime] = 0.3
    long_weight[low_vol_regime] = 0.6
    
    # Final Alpha Factor Construction
    combined_momentum = (short_weight * short_norm_momentum + 
                        medium_weight * medium_norm_momentum + 
                        long_weight * long_norm_momentum)
    
    # Multiply by volume divergence and apply smoothing
    alpha_factor = combined_momentum * volume_divergence
    alpha_factor_smoothed = alpha_factor.rolling(window=3, min_periods=1).mean()
    
    return alpha_factor_smoothed
