import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Range Persistence with Volume Confirmation
    # Calculate daily price range
    daily_range = data['high'] - data['low']
    
    # Compute range persistence
    range_ratio_3d = daily_range / daily_range.shift(1)
    range_momentum_5d = daily_range / daily_range.rolling(window=5, min_periods=3).mean()
    
    # Volume confirmation
    volume_slope_3d = data['volume'].rolling(window=3, min_periods=2).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0
    )
    volume_weighted_range = daily_range * data['volume']
    volume_weighted_range_persistence = volume_weighted_range / volume_weighted_range.rolling(window=5, min_periods=3).mean()
    
    range_persistence_factor = (range_ratio_3d + range_momentum_5d + volume_weighted_range_persistence) / 3
    
    # Volatility-Regime Adjusted Momentum
    # Identify volatility regime
    returns = data['close'].pct_change()
    vol_10d = returns.rolling(window=10, min_periods=8).std()
    vol_20d = returns.rolling(window=20, min_periods=15).std()
    vol_ratio = vol_10d / vol_20d
    
    # Regime classification
    high_vol_regime = (vol_ratio > 1.2).astype(int)
    low_vol_regime = (vol_ratio < 0.8).astype(int)
    
    # Compute regime-specific momentum
    intraday_momentum = (data['close'] - data['open']) / data['open']
    returns_2d = data['close'].pct_change(2)
    returns_5d_smooth = data['close'].pct_change(5).rolling(window=3, min_periods=2).mean()
    
    trend_persistence = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] > x.iloc[i-1]]) / (len(x)-1)
    )
    
    high_vol_momentum = returns_2d + intraday_momentum
    low_vol_momentum = returns_5d_smooth + trend_persistence
    
    volatility_regime_factor = high_vol_regime * high_vol_momentum + low_vol_regime * low_vol_momentum
    
    # Amount-Based Order Flow Imbalance
    # Calculate microstructure signals
    # Assuming daily data - using first and second half of day proxies
    morning_amount = data['amount'].rolling(window=2, min_periods=2).apply(lambda x: x.iloc[0])
    afternoon_amount = data['amount'].rolling(window=2, min_periods=2).apply(lambda x: x.iloc[1])
    session_ratio = afternoon_amount / morning_amount
    
    # Large trade concentration (simplified using rolling percentiles)
    large_trade_threshold = data['amount'].rolling(window=10, min_periods=8).quantile(0.9)
    large_trade_concentration = (data['amount'] > large_trade_threshold).astype(int)
    
    # Generate order flow imbalance factor
    order_flow_strength = session_ratio * large_trade_concentration
    trade_size_persistence = data['amount'].pct_change().rolling(window=3, min_periods=2).std()
    
    order_flow_factor = order_flow_strength - trade_size_persistence
    
    # Close-Relative Position Strength
    # Calculate position metrics
    position_daily = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    position_3d_avg = position_daily.rolling(window=3, min_periods=2).mean()
    position_trend = position_daily.rolling(window=3, min_periods=2).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / 3
    )
    
    # Generate predictive signal
    position_momentum = position_daily - position_3d_avg
    context_adjusted_strength = position_momentum * (1 + position_trend)
    
    position_factor = context_adjusted_strength
    
    # Volume-Price Divergence Detection
    # Calculate divergence components
    price_momentum_close = data['close'].pct_change(3)
    price_momentum_intraday = (data['close'] - data['open']) / data['open']
    
    volume_trend = data['volume'].pct_change(3)
    volume_acceleration = data['volume'].pct_change().rolling(window=3, min_periods=2).mean()
    
    # Detect divergence patterns
    positive_divergence = ((price_momentum_close < 0) & (volume_trend > 0)).astype(int)
    negative_divergence = ((price_momentum_close > 0) & (volume_trend < 0)).astype(int)
    
    divergence_factor = positive_divergence - negative_divergence
    
    # Combine all factors with equal weights
    combined_factor = (
        range_persistence_factor.fillna(0) + 
        volatility_regime_factor.fillna(0) + 
        order_flow_factor.fillna(0) + 
        position_factor.fillna(0) + 
        divergence_factor.fillna(0)
    ) / 5
    
    return combined_factor
