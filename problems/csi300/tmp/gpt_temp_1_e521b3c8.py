import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Divergence with Volume Confirmation
    # Calculate Short-Term Momentum (5-day ROC)
    short_momentum = df['close'].pct_change(periods=5)
    
    # Calculate Medium-Term Momentum (20-day ROC)
    medium_momentum = df['close'].pct_change(periods=20)
    
    # Compute Divergence (Short-Term minus Medium-Term)
    momentum_divergence = short_momentum - medium_momentum
    
    # Apply Volume Filter (multiply by current/5-day avg volume ratio)
    volume_ratio = df['volume'] / df['volume'].rolling(window=5).mean()
    momentum_factor = momentum_divergence * volume_ratio
    
    # Volatility Regime Adjusted Price Range
    # Calculate True Range (daily)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Determine Volatility Regime (20-day std dev vs historical median)
    vol_20day = df['close'].pct_change().rolling(window=20).std()
    vol_median = vol_20day.expanding().median()
    vol_regime = np.where(vol_20day > vol_median, -1, 1)
    
    # Adjust Range Signal (invert in high vol, amplify in low vol)
    range_signal = (df['close'] - df['open']) / true_range
    adjusted_range = range_signal * vol_regime
    
    # Liquidity-Adjusted Order Flow Imbalance
    # Calculate Tick-Based Direction (price movement signs)
    price_change = df['close'] - df['close'].shift(1)
    direction = np.sign(price_change)
    direction = direction.replace(0, 1)  # Treat zero changes as up
    
    # Compute Volume-Weighted Imbalance (direction × volume)
    raw_imbalance = direction * df['volume']
    
    # Adjust for Market Depth (scale by avg trade size)
    avg_trade_size = df['amount'] / df['volume']
    avg_trade_size_5day = avg_trade_size.rolling(window=5).mean()
    depth_adjustment = avg_trade_size / avg_trade_size_5day
    imbalance_factor = raw_imbalance * depth_adjustment
    
    # Volume-Weighted Price Level Clustering
    # Identify Support/Resistance Zones (high-volume price clusters)
    price_bins = pd.cut(df['close'], bins=20, labels=False)
    volume_by_price = df.groupby(price_bins)['volume'].transform('sum')
    high_volume_zones = volume_by_price > volume_by_price.rolling(window=20).quantile(0.7)
    
    # Calculate Current Price Position (distance to clusters)
    cluster_centers = df.groupby(price_bins)['close'].transform('mean')
    price_distance = (df['close'] - cluster_centers) / df['close'].rolling(window=20).std()
    
    # Generate Breakout/Pullback Signal (amplify near clusters)
    cluster_signal = np.where(high_volume_zones, price_distance * 2, price_distance * 0.5)
    
    # Relative Strength Rotation
    # Calculate Sector-Relative Performance (vs sector index)
    # Using rolling mean as proxy for sector index
    sector_proxy = df['close'].rolling(window=30).mean()
    relative_performance = df['close'] / sector_proxy - 1
    
    # Identify Rotation Patterns (relative strength shifts)
    rel_strength_5d = relative_performance.rolling(window=5).mean()
    rel_strength_20d = relative_performance.rolling(window=20).mean()
    rotation_momentum = rel_strength_5d - rel_strength_20d
    
    # Generate Rotation Signal (fade extremes, amplify early signs)
    rotation_zscore = rotation_momentum / rotation_momentum.rolling(window=50).std()
    rotation_signal = np.where(abs(rotation_zscore) > 2, -rotation_zscore * 0.5, rotation_zscore * 1.5)
    
    # Combine all factors with equal weights
    final_factor = (
        momentum_factor.fillna(0) * 0.2 +
        adjusted_range.fillna(0) * 0.2 +
        imbalance_factor.fillna(0) * 0.2 +
        cluster_signal.fillna(0) * 0.2 +
        rotation_signal.fillna(0) * 0.2
    )
    
    return final_factor
