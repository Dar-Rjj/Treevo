import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Volatility-Normalized Momentum Divergence with Volume Acceleration factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Momentum Component
    # Short-term momentum
    data['M_short_5'] = data['close'] / data['close'].shift(5) - 1
    data['M_short_10'] = data['close'] / data['close'].shift(10) - 1
    
    # Medium-term momentum
    data['M_medium_15'] = data['close'] / data['close'].shift(15) - 1
    data['M_medium_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Divergence
    data['short_div'] = data['M_short_5'] - data['M_short_10']
    data['medium_div'] = data['M_medium_15'] - data['M_medium_20']
    data['combined_div'] = (data['short_div'] + data['medium_div']) / 2
    
    # Volatility Normalization
    # Daily true range
    data['prev_close'] = data['close'].shift(1)
    data['TR1'] = data['high'] - data['low']
    data['TR2'] = abs(data['high'] - data['prev_close'])
    data['TR3'] = abs(data['low'] - data['prev_close'])
    data['TR'] = data[['TR1', 'TR2', 'TR3']].max(axis=1)
    
    # Rolling volatility
    data['vol_10d'] = data['TR'].rolling(window=10, min_periods=5).mean()
    data['vol_20d'] = data['TR'].rolling(window=20, min_periods=10).mean()
    
    # Adaptive volatility scaling
    data['adaptive_vol'] = data[['vol_10d', 'vol_20d']].min(axis=1)
    data['adaptive_vol'] = data['adaptive_vol'].clip(lower=1e-6)
    
    # Volume Acceleration Component
    # Volume momentum
    data['vol_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['vol_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['vol_momentum'] = data['vol_5d'] / data['vol_20d_avg']
    data['vol_momentum'] = data['vol_momentum'].clip(lower=0.5, upper=2.0)
    
    # Volume trend using linear regression slope
    def calc_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    data['vol_slope'] = data['volume'].rolling(window=5, min_periods=3).apply(
        calc_slope, raw=False
    )
    data['vol_trend'] = data['vol_slope'] / data['vol_20d_avg']
    
    # Factor Construction
    # Volatility-normalized momentum
    data['vol_norm_momentum'] = data['combined_div'] / data['adaptive_vol']
    # Preserve sign from short-term momentum
    data['vol_norm_momentum'] = np.sign(data['M_short_5']) * abs(data['vol_norm_momentum'])
    
    # Volume acceleration multiplier
    data['vol_acceleration'] = (data['vol_momentum'] + data['vol_trend']) / 2
    data['vol_multiplier'] = data['vol_acceleration'].clip(lower=0.8, upper=1.2)
    
    # Final factor assembly
    data['factor'] = data['vol_norm_momentum'] * data['vol_multiplier']
    
    # Robustness Enhancements
    # Outlier handling with sign preservation
    def winsorize_preserve_sign(series):
        abs_series = abs(series)
        threshold = abs_series.quantile(0.95)
        scaled = np.where(abs_series > threshold, 
                         threshold * np.sign(series), 
                         series)
        return scaled
    
    data['factor'] = winsorize_preserve_sign(data['factor'])
    
    # Market regime adaptation
    data['vol_regime'] = data['vol_20d'] / data['vol_20d'].rolling(window=60, min_periods=30).mean()
    regime_multiplier = np.where(data['vol_regime'] > 1.2, 0.7, 
                                np.where(data['vol_regime'] < 0.8, 1.3, 1.0))
    data['factor'] = data['factor'] * regime_multiplier
    
    return data['factor']
