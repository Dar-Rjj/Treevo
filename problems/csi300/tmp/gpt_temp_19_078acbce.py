import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate True Range
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility Regime Classification
    data['TR_5d_avg'] = data['TR'].rolling(window=5).mean()
    data['TR_20d_avg'] = data['TR'].rolling(window=20).mean()
    data['vol_ratio'] = data['TR_5d_avg'] / data['TR_20d_avg']
    
    # Efficiency ratio
    data['max_high_3d'] = data['high'].rolling(window=3).max()
    data['min_low_3d'] = data['low'].rolling(window=3).min()
    data['close_change_3d'] = data['close'] - data['close'].shift(3)
    data['efficiency_ratio'] = data['close_change_3d'] / (data['max_high_3d'] - data['min_low_3d'])
    
    # Multi-Scale Momentum
    data['acceleration'] = (data['close'] - data['close'].shift(3)) - (data['close'].shift(2) - data['close'].shift(5))
    data['momentum_21d'] = data['close'] - data['close'].shift(21)
    
    # Price-Volume Divergence
    # Volume trend slope (10-day linear regression)
    def volume_slope(volume_series):
        if len(volume_series) < 10:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = stats.linregress(x, volume_series)
        return slope
    
    data['volume_slope'] = data['volume'].rolling(window=10).apply(volume_slope, raw=True)
    data['divergence'] = np.sign(data['acceleration']) * np.sign(data['volume_slope'])
    
    # Range Efficiency
    data['range_efficiency'] = data['amount'] / (data['high'] - data['low'])
    
    # Adaptive Signal Synthesis
    # Define volatility regimes
    high_vol_threshold = data['vol_ratio'].quantile(0.7)
    low_vol_threshold = data['vol_ratio'].quantile(0.3)
    
    # Initialize factor
    factor = pd.Series(index=data.index, dtype=float)
    
    # High Volatility regime
    high_vol_mask = data['vol_ratio'] >= high_vol_threshold
    factor[high_vol_mask] = (
        data.loc[high_vol_mask, 'momentum_21d'] * 
        data.loc[high_vol_mask, 'efficiency_ratio']
    )
    
    # Low Volatility regime
    low_vol_mask = data['vol_ratio'] <= low_vol_threshold
    factor[low_vol_mask] = (
        data.loc[low_vol_mask, 'acceleration'] * 
        data.loc[low_vol_mask, 'divergence'] * 
        data.loc[low_vol_mask, 'range_efficiency']
    )
    
    # Normal Volatility regime
    normal_vol_mask = ~high_vol_mask & ~low_vol_mask
    factor[normal_vol_mask] = (
        data.loc[normal_vol_mask, 'acceleration'] * 
        data.loc[normal_vol_mask, 'divergence'] * 
        data.loc[normal_vol_mask, 'momentum_21d']
    )
    
    return factor
