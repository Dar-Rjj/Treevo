import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Cross-Asset Microstructure Momentum factor
    Combines cross-asset momentum patterns, microstructure efficiency, and multi-timeframe regime analysis
    """
    # Calculate basic price and volume features
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['price_momentum'] = df['close'] / df['close'].shift(5) - 1
    
    # Cross-asset momentum components (using rolling correlations as proxy)
    df['cross_asset_price_mom'] = df['returns'].rolling(window=10).corr(df['volume_change'].shift(1)).fillna(0)
    df['flow_momentum'] = (df['volume'] * df['returns']).rolling(window=5).mean()
    
    # Volatility transmission momentum
    df['volatility'] = df['returns'].rolling(window=5).std()
    df['vol_transmission_mom'] = df['volatility'].pct_change().rolling(window=3).mean()
    
    # Microstructure efficiency metrics
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['liquidity_momentum'] = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
    
    # Multi-timeframe regime framework
    # Ultra-short term (intraday) momentum regime
    df['intraday_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6)
    df['momentum_regime'] = df['intraday_momentum'].rolling(window=3).apply(
        lambda x: 1 if (x > 0).sum() >= 2 else (-1 if (x < 0).sum() >= 2 else 0)
    )
    
    # Short-term regime dynamics (1-3 days)
    df['regime_persistence'] = df['momentum_regime'].rolling(window=5).apply(
        lambda x: len(set(x)) if len(x) == 5 else 1
    )
    
    # Cross-timeframe alignment
    df['multi_timeframe_alignment'] = (
        df['momentum_regime'].rolling(window=3).std() / 
        df['price_momentum'].rolling(window=3).std().replace(0, 1e-6)
    )
    
    # Core momentum factors
    df['primary_alpha'] = (
        df['cross_asset_price_mom'] * 
        df['flow_momentum'].rolling(window=3).mean()
    )
    
    df['secondary_alpha'] = (
        df['vol_transmission_mom'] * 
        (1 - df['regime_persistence'] / 5)  # Regime quality confirmation
    )
    
    df['tertiary_alpha'] = (
        df['price_efficiency'].rolling(window=3).mean() * 
        df['liquidity_momentum']
    )
    
    # Hierarchical momentum composite
    weights = [0.4, 0.35, 0.25]  # Primary, secondary, tertiary weights
    df['hierarchical_momentum'] = (
        weights[0] * df['primary_alpha'] +
        weights[1] * df['secondary_alpha'] +
        weights[2] * df['tertiary_alpha']
    )
    
    # Cross-asset efficiency multiplier
    df['efficiency_multiplier'] = (
        df['multi_timeframe_alignment'].rolling(window=5).mean() *
        df['price_efficiency'].rolling(window=3).mean()
    )
    
    # Risk-adjusted momentum enhancement
    df['signal_noise_ratio'] = (
        df['hierarchical_momentum'].rolling(window=10).mean() / 
        df['hierarchical_momentum'].rolling(window=10).std().replace(0, 1e-6)
    )
    
    df['volatility_adjusted_mom'] = (
        df['hierarchical_momentum'] / 
        df['volatility'].rolling(window=5).mean().replace(0, 1e-6)
    )
    
    # Quality-weighted final alpha
    quality_weight = np.tanh(df['signal_noise_ratio'] * 0.1)  # Bound between -1 and 1
    df['quality_weighted_momentum'] = df['volatility_adjusted_mom'] * quality_weight
    
    # Final risk-adjusted alpha with efficiency enhancement
    alpha = (
        df['quality_weighted_momentum'] * 
        (1 + df['efficiency_multiplier'].clip(-0.5, 0.5))
    )
    
    return alpha.fillna(0)
