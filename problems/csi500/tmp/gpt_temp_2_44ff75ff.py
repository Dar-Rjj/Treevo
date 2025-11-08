import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining volatility-adjusted momentum,
    volume-price divergence, and intraday strength signals.
    """
    # Volatility-Adjusted Return Momentum (40%)
    # Calculate returns
    short_return = df['close'] / df['close'].shift(5) - 1
    medium_return = df['close'] / df['close'].shift(20) - 1
    
    # Calculate volatility measures
    high_low_range = (df['high'] - df['low']) / df['close']
    returns_20d = df['close'].pct_change()
    vol_std = returns_20d.rolling(window=20, min_periods=10).std()
    
    # Combine volatility measures
    volatility = (high_low_range + vol_std.fillna(0)) / 2
    
    # Volatility-adjusted returns
    vol_adj_short = short_return / (volatility + 1e-8)
    vol_adj_medium = medium_return / (volatility + 1e-8)
    
    # Weighted combination
    volatility_momentum = 0.6 * vol_adj_short + 0.4 * vol_adj_medium
    
    # Volume-Price Divergence Factor (35%)
    # Price trends
    price_5d_slope = df['close'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == 5 else np.nan, 
        raw=False
    )
    price_20d_slope = df['close'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Volume trends
    volume_5d_slope = df['volume'].rolling(window=5).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-8) if len(x) == 5 else np.nan, 
        raw=False
    )
    volume_20d_slope = df['volume'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-8) if len(x) == 20 else np.nan, 
        raw=False
    )
    
    # Divergence signals
    short_divergence = price_5d_slope - volume_5d_slope
    medium_divergence = price_20d_slope - volume_20d_slope
    
    # Combine divergence signals
    volume_divergence = 0.7 * short_divergence + 0.3 * medium_divergence
    
    # Intraday Strength Indicator (25%)
    # Daily price efficiency
    daily_efficiency = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    efficiency_5d = daily_efficiency.rolling(window=5, min_periods=3).mean()
    
    # Volume-weighted price position
    price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    volume_weighted_position = price_position * df['volume']
    
    # Normalize volume-weighted position
    volume_weighted_position_norm = (
        volume_weighted_position - volume_weighted_position.rolling(window=20).mean()
    ) / (volume_weighted_position.rolling(window=20).std() + 1e-8)
    
    # Combine intraday signals
    intraday_signal = efficiency_5d * volume_weighted_position_norm
    intraday_strength = intraday_signal - intraday_signal.shift(5)
    
    # Final Alpha Factor Integration
    # Normalize components
    vol_mom_norm = (
        volatility_momentum - volatility_momentum.rolling(window=20).mean()
    ) / (volatility_momentum.rolling(window=20).std() + 1e-8)
    
    vol_div_norm = (
        volume_divergence - volume_divergence.rolling(window=20).mean()
    ) / (volume_divergence.rolling(window=20).std() + 1e-8)
    
    intra_norm = (
        intraday_strength - intraday_strength.rolling(window=20).mean()
    ) / (intraday_strength.rolling(window=20).std() + 1e-8)
    
    # Weighted combination
    composite_alpha = (
        0.4 * vol_mom_norm + 
        0.35 * vol_div_norm + 
        0.25 * intra_norm
    )
    
    # Apply smoothing
    final_alpha = composite_alpha.rolling(window=5, min_periods=3).mean()
    
    return final_alpha
