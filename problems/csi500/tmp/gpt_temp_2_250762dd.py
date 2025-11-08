import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price Path Fractality with Volume-Value Congruence factor
    Combines multi-scale fractal dimension analysis with volume-value distribution patterns
    """
    data = df.copy()
    
    # 1. Multi-Scale Fractal Dimension Analysis
    # Short-term fractal complexity (3-day)
    data['high_3d'] = data['high'].rolling(window=3, min_periods=3).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=3).min()
    data['price_change_sum_3d'] = data['close'].diff().abs().rolling(window=3, min_periods=3).sum()
    data['fractal_3d'] = (data['high_3d'] - data['low_3d']) / (data['price_change_sum_3d'] + 1e-8)
    
    # 5-day path irregularity
    data['high_5d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['range_sum_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=5).sum()
    data['fractal_5d'] = (data['high_5d'] - data['low_5d']) / (data['range_sum_5d'] + 1e-8)
    
    # Fractal persistence patterns
    data['fractal_3d_change'] = data['fractal_3d'].diff()
    data['fractal_persistence'] = data['fractal_3d_change'].rolling(window=3, min_periods=3).apply(
        lambda x: np.sum(x > 0) if len(x) == 3 else np.nan
    )
    
    # Medium-term scaling properties (10-20 day fractal ratio)
    data['high_10d'] = data['high'].rolling(window=10, min_periods=10).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=10).min()
    data['price_change_sum_10d'] = data['close'].diff().abs().rolling(window=10, min_periods=10).sum()
    data['fractal_10d'] = (data['high_10d'] - data['low_10d']) / (data['price_change_sum_10d'] + 1e-8)
    
    data['high_20d'] = data['high'].rolling(window=20, min_periods=20).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=20).min()
    data['price_change_sum_20d'] = data['close'].diff().abs().rolling(window=20, min_periods=20).sum()
    data['fractal_20d'] = (data['high_20d'] - data['low_20d']) / (data['price_change_sum_20d'] + 1e-8)
    
    data['fractal_ratio_10_20'] = data['fractal_10d'] / (data['fractal_20d'] + 1e-8)
    
    # 2. Volume-Value Distribution Analysis
    # High-Low Volume Asymmetry (10-day)
    data['price_change'] = data['close'].pct_change()
    data['up_day_volume'] = np.where(data['price_change'] > 0, data['volume'], 0)
    data['down_day_volume'] = np.where(data['price_change'] < 0, data['volume'], 0)
    
    data['up_volume_10d'] = data['up_day_volume'].rolling(window=10, min_periods=10).sum()
    data['down_volume_10d'] = data['down_day_volume'].rolling(window=10, min_periods=10).sum()
    data['total_volume_10d'] = data['volume'].rolling(window=10, min_periods=10).sum()
    
    data['volume_asymmetry'] = (data['up_volume_10d'] - data['down_volume_10d']) / (data['total_volume_10d'] + 1e-8)
    
    # Volume cluster persistence
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['high_volume_days'] = (data['volume'] > data['volume_ma_5d']).rolling(window=5, min_periods=5).sum()
    
    # Volume distribution skewness
    data['volume_skew_10d'] = data['volume'].rolling(window=10, min_periods=10).skew()
    
    # 3. Fractal-Volume Divergence Dynamics
    # Short-term fractal-volume misalignment
    data['volume_ma_3d'] = data['volume'].rolling(window=3, min_periods=3).mean()
    data['volume_zscore_3d'] = (data['volume'] - data['volume_ma_3d']) / (data['volume'].rolling(window=3, min_periods=3).std() + 1e-8)
    
    data['fractal_volume_divergence'] = data['fractal_3d'] * data['volume_zscore_3d']
    
    # Volume-weighted price efficiency
    data['vwap'] = (data['amount'] / data['volume']).replace([np.inf, -np.inf], np.nan)
    data['price_efficiency'] = (data['close'] - data['vwap']).abs() / data['close']
    data['volume_weighted_efficiency'] = data['price_efficiency'] * data['volume_zscore_3d']
    
    # 4. Combined Factor Construction
    # Normalize components
    components = ['fractal_3d', 'fractal_5d', 'fractal_ratio_10_20', 'volume_asymmetry', 
                 'fractal_volume_divergence', 'volume_weighted_efficiency']
    
    # Calculate z-scores for each component
    factor_components = pd.DataFrame()
    for comp in components:
        if comp in data.columns:
            factor_components[comp] = (data[comp] - data[comp].rolling(window=20, min_periods=20).mean()) / \
                                    (data[comp].rolling(window=20, min_periods=20).std() + 1e-8)
    
    # Weighted combination based on theoretical importance
    weights = {
        'fractal_3d': 0.25,
        'fractal_5d': 0.20,
        'fractal_ratio_10_20': 0.15,
        'volume_asymmetry': 0.15,
        'fractal_volume_divergence': 0.15,
        'volume_weighted_efficiency': 0.10
    }
    
    # Calculate final factor
    factor = pd.Series(index=data.index, dtype=float)
    for comp, weight in weights.items():
        if comp in factor_components.columns:
            factor = factor.add(factor_components[comp] * weight, fill_value=0)
    
    # Apply smoothing and remove extreme outliers
    factor = factor.rolling(window=3, min_periods=3).mean()
    factor = np.clip(factor, factor.quantile(0.01), factor.quantile(0.99))
    
    return factor
