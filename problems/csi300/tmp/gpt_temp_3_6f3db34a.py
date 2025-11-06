import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Short-term Momentum (2-day)
    data['high_2d'] = data['high'].rolling(window=3).max()
    data['low_2d'] = data['low'].rolling(window=3).min()
    data['short_momentum'] = (data['close'] - data['close'].shift(2)) / (data['high_2d'] - data['low_2d']).replace(0, np.nan)
    
    # Medium-term Momentum (6-day)
    data['high_6d'] = data['high'].rolling(window=7).max()
    data['low_6d'] = data['low'].rolling(window=7).min()
    data['medium_momentum'] = (data['close'] - data['close'].shift(6)) / (data['high_6d'] - data['low_6d']).replace(0, np.nan)
    
    # Price Efficiency Assessment
    # Efficiency Ratio
    close_diff = data['close'].diff().abs()
    data['efficiency_ratio'] = (data['close'] - data['close'].shift(5)) / close_diff.rolling(window=5).sum().replace(0, np.nan)
    
    # Volatility Clustering
    data['volatility_clustering'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5)).replace(0, np.nan)
    
    # Volume-Price Integration
    # Volume Persistence
    data['volume_persistence'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    
    # Price-Volume Divergence
    data['price_volume_divergence'] = ((data['close'] - data['close'].shift(1)) / data['volume']) - \
                                     ((data['close'].shift(1) - data['close'].shift(2)) / data['volume'].shift(1))
    
    # Volume Concentration
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=4).mean()
    
    # Signal Confirmation
    # Momentum-Efficiency Alignment
    data['momentum_efficiency_alignment'] = data['intraday_momentum'] * data['efficiency_ratio']
    
    # Volume-Price Consistency
    momentum_direction = np.sign(data['close'] - data['close'].shift(1))
    volume_direction = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_price_consistency'] = momentum_direction * volume_direction * data['volume_persistence']
    
    # Volatility Context
    data['volatility_context'] = data['volatility_clustering'] / data['volatility_clustering'].rolling(window=10).mean()
    
    # Composite Alpha Generation
    # Normalize components
    momentum_components = pd.concat([
        data['intraday_momentum'],
        data['short_momentum'],
        data['medium_momentum']
    ], axis=1).apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    efficiency_components = pd.concat([
        data['efficiency_ratio'],
        data['momentum_efficiency_alignment']
    ], axis=1).apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    volume_components = pd.concat([
        data['volume_persistence'],
        data['price_volume_divergence'],
        data['volume_concentration'],
        data['volume_price_consistency']
    ], axis=1).apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    # Regime-adaptive weighting
    volatility_regime = np.where(data['volatility_context'] > 1, 1.2, 0.8)
    
    # Weighted combination
    momentum_score = momentum_components.mean(axis=1)
    efficiency_score = efficiency_components.mean(axis=1)
    volume_score = volume_components.mean(axis=1)
    
    # Final alpha factor with regime-adaptive weights
    alpha = (volatility_regime * 0.4 * momentum_score + 
             0.35 * efficiency_score + 
             0.25 * volume_score)
    
    # Apply directional bias from momentum-efficiency alignment
    alignment_strength = data['momentum_efficiency_alignment'].abs()
    alignment_direction = np.sign(data['momentum_efficiency_alignment'])
    
    # Enhanced alpha with momentum-efficiency confirmation
    enhanced_alpha = alpha * (1 + 0.3 * alignment_strength * alignment_direction)
    
    return enhanced_alpha
