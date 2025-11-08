import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum with Volume Acceleration factor
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Framework
    df['ultra_short_mom'] = df['close'] / df['close'].shift(2) - 1
    df['short_mom'] = df['close'] / df['close'].shift(5) - 1
    df['medium_mom'] = df['close'] / df['close'].shift(10) - 1
    df['long_mom'] = df['close'] / df['close'].shift(20) - 1
    
    # Calculate daily returns for volatility
    df['daily_ret'] = df['close'] / df['close'].shift(1) - 1
    
    # Dynamic Volatility Regime Detection
    df['vol_5d'] = df['daily_ret'].rolling(window=5).std()
    df['vol_15d'] = df['daily_ret'].rolling(window=15).std()
    df['vol_30d'] = df['daily_ret'].rolling(window=30).std()
    
    # Regime Classification
    conditions = [
        (df['vol_5d'] > df['vol_15d']) & (df['vol_5d'] > df['vol_30d']),  # High Volatility
        (df['vol_15d'] > df['vol_5d']) & (df['vol_15d'] > df['vol_30d']),  # Medium Volatility
        (df['vol_30d'] > df['vol_5d']) & (df['vol_30d'] > df['vol_15d'])   # Low Volatility
    ]
    choices = ['high', 'medium', 'low']
    df['vol_regime'] = np.select(conditions, choices, default='medium')
    
    # Volume Acceleration Analysis
    df['volume_roc_3d'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_roc_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_roc_10d'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_acceleration'] = df['volume_roc_5d'] - df['volume_roc_10d']
    
    # Volume Trend Persistence
    df['sma_volume_10'] = df['volume'].rolling(window=10).mean().shift(1)
    df['volume_above_sma'] = df['volume'] > df['sma_volume_10']
    
    # Calculate consecutive days with volume above SMA
    df['volume_trend_persistence'] = 0
    for i in range(1, len(df)):
        if df['volume_above_sma'].iloc[i]:
            df['volume_trend_persistence'].iloc[i] = df['volume_trend_persistence'].iloc[i-1] + 1
        else:
            df['volume_trend_persistence'].iloc[i] = 0
    
    # Price-Volume Divergence
    df['momentum_volume_alignment'] = np.sign(df['ultra_short_mom']) * np.sign(df['volume_roc_3d'])
    
    # Calculate z-scores for divergence strength
    df['mom_zscore'] = (df['ultra_short_mom'] - df['ultra_short_mom'].rolling(window=20).mean()) / df['ultra_short_mom'].rolling(window=20).std()
    df['vol_zscore'] = (df['volume_roc_3d'] - df['volume_roc_3d'].rolling(window=20).mean()) / df['volume_roc_3d'].rolling(window=20).std()
    df['divergence_strength'] = abs(df['mom_zscore'] - df['vol_zscore'])
    
    # Volume Regime Classification
    df['sma_volume_20'] = df['volume'].rolling(window=20).mean().shift(1)
    volume_conditions = [
        (df['volume'] > df['sma_volume_20']) & (df['volume_roc_3d'] > 0),  # High Volume
        (df['volume'] >= df['sma_volume_10']) & (df['volume'] <= df['sma_volume_20']),  # Normal Volume
        (df['volume'] < df['sma_volume_10']) & (df['volume_roc_3d'] < 0)   # Low Volume
    ]
    volume_choices = ['high', 'normal', 'low']
    df['volume_regime'] = np.select(volume_conditions, volume_choices, default='normal')
    
    # Adaptive Alpha Construction
    # Regime-Weighted Momentum Selection
    def get_regime_weighted_momentum(row):
        if row['vol_regime'] == 'high':
            return 0.7 * row['ultra_short_mom'] + 0.3 * row['short_mom']
        elif row['vol_regime'] == 'medium':
            return 0.4 * row['short_mom'] + 0.4 * row['medium_mom'] + 0.2 * row['long_mom']
        else:  # low volatility
            return 0.2 * row['medium_mom'] + 0.8 * row['long_mom']
    
    df['selected_momentum'] = df.apply(get_regime_weighted_momentum, axis=1)
    
    # Volume Acceleration Multiplier
    df['volume_acceleration_score'] = df['volume_roc_3d'] * df['volume_trend_persistence']
    df['base_factor'] = df['selected_momentum'] * df['volume_acceleration_score']
    
    # Divergence-Based Adjustment
    def apply_divergence_adjustment(row):
        if row['momentum_volume_alignment'] < 0 and row['divergence_strength'] > 1:
            return row['base_factor'] * 0.5  # Strong Divergence
        elif row['momentum_volume_alignment'] > 0 and row['divergence_strength'] > 1:
            return row['base_factor'] * 1.25  # Strong Alignment
        else:
            return row['base_factor']  # Normal Alignment
    
    df['factor'] = df.apply(apply_divergence_adjustment, axis=1)
    
    return df['factor']
