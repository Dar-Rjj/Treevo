import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Momentum Divergence with Volume-Price Efficiency factor
    """
    # Make copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Divergence Detection
    # 3-day and 10-day momentum calculation
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Momentum Divergence Measurement
    data['momentum_divergence'] = np.where(
        abs(data['momentum_10d']) > 0.001,  # Avoid division by zero
        (data['momentum_3d'] - data['momentum_10d']) / abs(data['momentum_10d']),
        0
    )
    
    # Divergence Persistence Analysis
    data['positive_divergence'] = (data['momentum_divergence'] > 0).astype(int)
    data['divergence_persistence'] = data['positive_divergence'].rolling(window=5, min_periods=1).sum()
    data['persistence_strength'] = np.where(
        data['divergence_persistence'] >= 2,
        data['divergence_persistence'] / 5,
        0
    )
    
    # Volume-Price Efficiency Assessment
    # Daily price efficiency ratio
    data['daily_range'] = data['high'] - data['low']
    data['price_efficiency'] = np.where(
        data['daily_range'] > 0,
        (data['close'] - data['open']) / data['daily_range'],
        0
    )
    
    # 5-day efficiency trend
    data['efficiency_ma_5d'] = data['price_efficiency'].rolling(window=5, min_periods=1).mean()
    data['efficiency_trend'] = data['price_efficiency'] - data['efficiency_ma_5d'].shift(1)
    
    # Volume Confirmation Analysis
    data['up_day'] = (data['close'] > data['open']).astype(int)
    data['down_day'] = (data['close'] < data['open']).astype(int)
    
    # 5-day volume asymmetry
    up_volume = data['volume'] * data['up_day']
    down_volume = data['volume'] * data['down_day']
    data['volume_asymmetry'] = (
        up_volume.rolling(window=5, min_periods=1).mean() - 
        down_volume.rolling(window=5, min_periods=1).mean()
    ) / data['volume'].rolling(window=5, min_periods=1).mean()
    
    # Volume-efficiency correlation (5-day rolling)
    data['volume_efficiency_corr'] = (
        data['volume'].rolling(window=5, min_periods=1).corr(data['price_efficiency'])
    )
    
    # Range-Based Volatility Context
    # 10-day historical range percentiles
    data['daily_range_pct'] = data['daily_range'].rolling(window=10, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0
    )
    
    # Range expansion/contraction detection
    data['median_range_10d'] = data['daily_range'].rolling(window=10, min_periods=1).median()
    data['range_regime'] = np.where(
        data['daily_range'] > data['median_range_10d'],
        1.2,  # Expansion phase multiplier
        0.8   # Contraction phase multiplier
    )
    
    # Multi-Timeframe Signal Integration
    # Temporal Pattern Recognition
    data['divergence_strength_3d'] = data['momentum_divergence'].rolling(window=3, min_periods=1).mean()
    
    # Volume-Efficiency Consistency
    data['efficiency_trend_5d'] = data['price_efficiency'].rolling(window=5, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) if len(x) == 5 else 0
    )
    
    # Composite Alpha Generation
    # Divergence-Efficiency Integration
    data['divergence_efficiency_score'] = (
        data['momentum_divergence'] * 
        np.where(data['price_efficiency'] > 0, 1 + data['price_efficiency'], 1)
    )
    
    # Apply volume confirmation overlay
    volume_confirmation = np.where(
        (data['volume_asymmetry'] * np.sign(data['momentum_divergence'])) > 0,
        1.2,  # Strong volume confirmation
        np.where(data['volume_efficiency_corr'] > 0.3, 1.1, 1.0)  # Moderate confirmation
    )
    
    data['volume_enhanced_divergence'] = data['divergence_efficiency_score'] * volume_confirmation
    
    # Volatility-Adaptive Signal Refinement
    data['volatility_adjusted_signal'] = (
        data['volume_enhanced_divergence'] * 
        data['range_regime'] * 
        data['persistence_strength']
    )
    
    # Final Predictive Output
    # Multi-timeframe momentum divergence factor with volume-price efficiency enhancement
    alpha_factor = (
        data['volatility_adjusted_signal'] * 
        (1 + data['efficiency_trend']) * 
        np.where(data['divergence_strength_3d'] > 0, 1.1, 1.0)
    )
    
    return alpha_factor
