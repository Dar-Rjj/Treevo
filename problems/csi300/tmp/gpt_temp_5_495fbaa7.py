import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Momentum Component
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Volume Divergence Component
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    data['price_volume_divergence'] = data['short_term_momentum'] - data['volume_momentum']
    
    # Range Efficiency Component
    data['daily_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['daily_range_efficiency'] = data['daily_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # 3-Day Range Persistence
    data['range_efficiency_gt_05'] = (data['daily_range_efficiency'] > 0.5).astype(int)
    data['range_persistence_3d'] = data['range_efficiency_gt_05'].rolling(window=3, min_periods=1).sum()
    
    # Factor Integration
    data['momentum_divergence_score'] = data['price_volume_divergence'] * data['daily_range_efficiency']
    
    # Volume Confirmation Filter
    data['volume_persistence'] = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['filtered_score'] = data['momentum_divergence_score'] * data['volume_persistence']
    
    # Cross-Sectional Ranking (within each day)
    def cross_sectional_rank(series):
        return series.rank(pct=True)
    
    data['cross_sectional_rank'] = data.groupby(data.index)['filtered_score'].transform(cross_sectional_rank)
    
    # Time-Series Smoothing
    data['final_factor'] = data['cross_sectional_rank'].rolling(window=3, min_periods=1).mean()
    
    return data['final_factor']
