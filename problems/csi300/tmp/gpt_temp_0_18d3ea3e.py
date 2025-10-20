import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Efficiency Metrics
    data['short_term_efficiency'] = (data['high'] - data['low']) / data['close']
    
    # Medium-term efficiency using rolling windows
    data['high_5d'] = data['high'].rolling(window=5, min_periods=1).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=1).min()
    data['medium_term_efficiency'] = (data['high_5d'] - data['low_5d']) / data['close']
    
    # Efficiency ratio
    data['efficiency_ratio'] = data['short_term_efficiency'] / data['medium_term_efficiency']
    data['efficiency_ratio'] = data['efficiency_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Price persistence
    data['close_prev1'] = data['close'].shift(1)
    data['close_prev2'] = data['close'].shift(2)
    data['price_persistence'] = ((data['close'] > data['close_prev1']) & 
                                (data['close_prev1'] > data['close_prev2'])).astype(float)
    
    # Volume Efficiency Patterns
    data['volume_sum_4d'] = data['volume'].rolling(window=4, min_periods=1).sum()
    data['volume_concentration'] = data['volume'] / data['volume_sum_4d']
    
    # Volume efficiency
    data['price_change_abs'] = abs(data['close'] - data['close_prev1'])
    data['volume_efficiency'] = data['price_change_abs'] / data['volume']
    data['volume_efficiency'] = data['volume_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Volume clustering
    data['volume_mean_prev3'] = data['volume'].shift(1).rolling(window=3, min_periods=1).mean()
    data['volume_mean_prev4'] = data['volume'].shift(2).rolling(window=3, min_periods=1).mean()
    data['volume_clustering'] = ((data['volume'] > data['volume_mean_prev3']) & 
                               (data['volume'].shift(1) > data['volume_mean_prev4'])).astype(float)
    
    # Volume-price efficiency (correlation)
    data['price_change_abs_5d'] = abs(data['close'] - data['close'].shift(1)).rolling(window=5, min_periods=1).apply(lambda x: x.tolist(), raw=False)
    data['volume_5d'] = data['volume'].rolling(window=5, min_periods=1).apply(lambda x: x.tolist(), raw=False)
    
    def calculate_corr(price_vol_data):
        if len(price_vol_data) == 2 and len(price_vol_data[0]) >= 2 and len(price_vol_data[1]) >= 2:
            price_changes = price_vol_data[0]
            volumes = price_vol_data[1]
            if len(price_changes) == len(volumes):
                return np.corrcoef(price_changes, volumes)[0, 1] if len(price_changes) > 1 else np.nan
        return np.nan
    
    data['volume_price_efficiency'] = data[['price_change_abs_5d', 'volume_5d']].apply(calculate_corr, axis=1)
    
    # Market Microstructure Regimes
    data['volume_mean_5d'] = data['volume'].shift(1).rolling(window=5, min_periods=1).mean()
    data['high_friction'] = ((data['high'] - data['low']) > 0.015 * data['close']) & \
                           (data['volume'] < 0.8 * data['volume_mean_5d'])
    
    data['price_change_8d_pct'] = abs(data['close'] / data['close'].shift(8) - 1)
    data['volume_efficiency_mean_5d'] = data['volume_efficiency'].shift(1).rolling(window=5, min_periods=1).mean()
    data['efficient_trending'] = (data['price_change_8d_pct'] > 0.04) & \
                                (data['volume_efficiency'] > data['volume_efficiency_mean_5d'])
    
    data['volume_compression'] = (data['volume_concentration'] > 0.3) & \
                                (data['volume_clustering'] == 1)
    
    # Alpha Generation based on regimes
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Friction regime alpha
    friction_alpha = -data['efficiency_ratio'] * data['volume_concentration']
    
    # Efficient trend regime alpha  
    trend_alpha = data['price_persistence'] * data['volume_efficiency']
    
    # Compression regime alpha
    compression_alpha = data['volume_price_efficiency'] * data['volume_clustering']
    
    # Combine based on dominant regime
    for idx in data.index:
        if data.loc[idx, 'high_friction']:
            alpha.loc[idx] = friction_alpha.loc[idx]
        elif data.loc[idx, 'efficient_trending']:
            alpha.loc[idx] = trend_alpha.loc[idx]
        elif data.loc[idx, 'volume_compression']:
            alpha.loc[idx] = compression_alpha.loc[idx]
        else:
            # Default: weighted average of all regimes
            alpha.loc[idx] = 0.4 * friction_alpha.loc[idx] + 0.4 * trend_alpha.loc[idx] + 0.2 * compression_alpha.loc[idx]
    
    # Clean up intermediate columns
    cols_to_drop = ['high_5d', 'low_5d', 'close_prev1', 'close_prev2', 'volume_sum_4d', 
                   'price_change_abs', 'volume_mean_prev3', 'volume_mean_prev4', 
                   'price_change_abs_5d', 'volume_5d', 'volume_mean_5d', 
                   'price_change_8d_pct', 'volume_efficiency_mean_5d']
    
    for col in cols_to_drop:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
    
    return alpha
