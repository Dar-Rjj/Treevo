import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum-Divergence Alpha Factor
    """
    data = df.copy()
    
    # Volatility-Efficiency Regime Classification
    # True Range calculation
    data['TR'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Volatility components
    data['TR_5d'] = data['TR'].rolling(window=5, min_periods=3).mean()
    data['TR_20d'] = data['TR'].rolling(window=20, min_periods=10).mean()
    data['vol_ratio'] = data['TR_5d'] / data['TR_20d']
    
    # Efficiency components
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['max_high_3d'] = data['high'].rolling(window=3, min_periods=2).max()
    data['min_low_3d'] = data['low'].rolling(window=3, min_periods=2).min()
    data['range_3d'] = data['max_high_3d'] - data['min_low_3d']
    data['efficiency_ratio'] = data['price_change_3d'] / (data['range_3d'] + 1e-8)
    
    # Multi-Scale Acceleration Momentum
    # Short-term acceleration
    data['momentum_3d'] = data['close'] - data['close'].shift(3)
    data['momentum_3d_lag2'] = data['close'].shift(2) - data['close'].shift(5)
    data['acceleration'] = data['momentum_3d'] - data['momentum_3d_lag2']
    
    # Medium-term momentum
    data['momentum_21d'] = data['close'] - data['close'].shift(21)
    
    # Price-Volume Divergence Analysis
    # Volume trend component
    def volume_slope(series):
        if len(series) < 3:
            return 0
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    data['volume_slope'] = data['volume'].rolling(window=10, min_periods=5).apply(
        volume_slope, raw=False
    )
    
    # Price-volume alignment
    data['divergence_indicator'] = np.where(
        (data['acceleration'] * data['volume_slope']) > 0, 1, -1
    )
    
    # Range efficiency confirmation
    data['daily_range'] = data['high'] - data['low']
    data['range_efficiency'] = data['amount'] / (data['daily_range'] + 1e-8)
    
    # Volatility regime classification
    data['vol_regime'] = np.select(
        [
            data['vol_ratio'] > 1.2,  # High volatility
            data['vol_ratio'] < 0.8,  # Low volatility
        ],
        [2, 0],  # High=2, Low=0
        default=1  # Normal=1
    )
    
    # Adaptive Signal Synthesis
    # High volatility regime (2)
    high_vol_mask = data['vol_regime'] == 2
    data.loc[high_vol_mask, 'momentum_weight'] = 0.7
    data.loc[high_vol_mask, 'divergence_weight'] = 0.2
    data.loc[high_vol_mask, 'efficiency_weight'] = 0.1
    data.loc[high_vol_mask, 'momentum_signal'] = data['momentum_21d']
    
    # Low volatility regime (0)
    low_vol_mask = data['vol_regime'] == 0
    data.loc[low_vol_mask, 'momentum_weight'] = 0.3
    data.loc[low_vol_mask, 'divergence_weight'] = 0.4
    data.loc[low_vol_mask, 'efficiency_weight'] = 0.3
    data.loc[low_vol_mask, 'momentum_signal'] = data['momentum_3d'].rolling(
        window=5, min_periods=3
    ).mean()
    
    # Normal volatility regime (1)
    normal_vol_mask = data['vol_regime'] == 1
    data.loc[normal_vol_mask, 'momentum_weight'] = 0.5
    data.loc[normal_vol_mask, 'divergence_weight'] = 0.3
    data.loc[normal_vol_mask, 'efficiency_weight'] = 0.2
    data.loc[normal_vol_mask, 'momentum_signal'] = (
        data['momentum_3d'] * 0.6 + data['momentum_21d'] * 0.4
    )
    
    # Final Factor Construction
    # Base signal components
    data['base_signal'] = (
        data['acceleration'] * 
        data['divergence_indicator'] * 
        data['efficiency_ratio']
    )
    
    # Regime-adaptive momentum overlay
    data['momentum_component'] = data['momentum_weight'] * data['momentum_signal']
    
    # Volume-confirmed divergence adjustment
    data['divergence_component'] = (
        data['divergence_weight'] * 
        data['divergence_indicator'] * 
        np.sign(data['volume_slope'])
    )
    
    # Efficiency component
    data['efficiency_component'] = (
        data['efficiency_weight'] * 
        data['range_efficiency'] * 
        np.sign(data['efficiency_ratio'])
    )
    
    # Final factor
    data['factor'] = (
        data['base_signal'] * 0.4 +
        data['momentum_component'] * 0.4 +
        data['divergence_component'] * 0.15 +
        data['efficiency_component'] * 0.05
    )
    
    # Clean and return
    factor_series = data['factor'].replace([np.inf, -np.inf], np.nan)
    return factor_series
