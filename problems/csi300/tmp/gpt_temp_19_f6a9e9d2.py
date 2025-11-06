import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Efficiency with Bidirectional Flow Analysis
    Multi-dimensional volume analysis focusing on volume fractal efficiency,
    bidirectional flow analysis, and price-level volume anchoring.
    """
    data = df.copy()
    
    # Multi-Timeframe Volume Fractals
    # Short-term Volume Clustering
    data['volume_var_3d'] = data['volume'].rolling(window=3).var()
    data['volume_clustered'] = (data['volume_var_3d'] < data['volume_var_3d'].rolling(window=20).quantile(0.3)).astype(int)
    
    # Medium-term Volume Persistence
    data['volume_autocorr_10d'] = data['volume'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) == 10 and not x.isna().any() else np.nan
    )
    data['volume_regime'] = (data['volume_autocorr_10d'] > data['volume_autocorr_10d'].rolling(window=50).quantile(0.7)).astype(int)
    
    # Volume-Price Efficiency Ratio
    data['daily_range'] = data['high'] - data['low']
    data['efficiency_ratio'] = data['daily_range'] / (data['volume'] + 1e-8)
    
    # Bidirectional Flow Analysis
    data['price_change'] = data['close'] - data['open']
    data['is_up_day'] = (data['price_change'] > 0).astype(int)
    data['is_down_day'] = (data['price_change'] < 0).astype(int)
    
    # Up vs Down Volume Efficiency
    up_days = data[data['is_up_day'] == 1].index
    down_days = data[data['is_down_day'] == 1].index
    
    data['up_efficiency'] = np.nan
    data['down_efficiency'] = np.nan
    
    if len(up_days) > 0:
        data.loc[up_days, 'up_efficiency'] = data.loc[up_days, 'daily_range'] / (data.loc[up_days, 'volume'] + 1e-8)
    if len(down_days) > 0:
        data.loc[down_days, 'down_efficiency'] = data.loc[down_days, 'daily_range'] / (data.loc[down_days, 'volume'] + 1e-8)
    
    data['up_efficiency'] = data['up_efficiency'].fillna(method='ffill')
    data['down_efficiency'] = data['down_efficiency'].fillna(method='ffill')
    data['flow_divergence'] = data['up_efficiency'] - data['down_efficiency']
    
    # Price-Level Volume Anchoring
    data['resistance_level'] = data['high'].rolling(window=5).max()
    data['support_level'] = data['low'].rolling(window=5).min()
    
    # Volume concentration near key levels
    data['near_resistance'] = ((data['high'] >= data['resistance_level'] * 0.995) & 
                              (data['high'] <= data['resistance_level'] * 1.005)).astype(int)
    data['near_support'] = ((data['low'] >= data['support_level'] * 0.995) & 
                           (data['low'] <= data['support_level'] * 1.005)).astype(int)
    
    data['resistance_volume'] = data['volume'] * data['near_resistance']
    data['support_volume'] = data['volume'] * data['near_support']
    
    # Flow-Momentum Asymmetry Detection
    data['up_volume_momentum'] = data['volume'].rolling(window=5).apply(
        lambda x: x[data['is_up_day'].iloc[-5:].values == 1].mean() if len(x) == 5 else np.nan
    )
    data['down_volume_momentum'] = data['volume'].rolling(window=5).apply(
        lambda x: x[data['is_down_day'].iloc[-5:].values == 1].mean() if len(x) == 5 else np.nan
    )
    
    # Volume-Price Fractal Correlation
    data['abs_price_change'] = abs(data['close'] - data['close'].shift(1))
    data['volume_price_corr_3d'] = data['volume'].rolling(window=3).corr(data['abs_price_change'])
    data['volume_price_corr_10d'] = data['volume'].rolling(window=10).corr(data['abs_price_change'])
    
    # Volume Regime Adaptive Weighting
    volume_variance_20d = data['volume'].rolling(window=20).var()
    data['volume_concentration_regime'] = (volume_variance_20d > volume_variance_20d.rolling(window=50).quantile(0.75)).astype(int)
    
    # Composite Volume Fractal Momentum
    # Core components with regime-adaptive weighting
    efficiency_component = data['efficiency_ratio'].rolling(window=5).mean()
    flow_component = data['flow_divergence'].rolling(window=5).mean()
    
    # Level-break confirmation component
    breakout_volume = data['resistance_volume'].rolling(window=3).mean()
    support_volume = data['support_volume'].rolling(window=3).mean()
    level_component = (breakout_volume - support_volume) / (data['volume'].rolling(window=10).mean() + 1e-8)
    
    # Asymmetry component
    volume_momentum_asymmetry = (data['up_volume_momentum'] - data['down_volume_momentum']) / (
        data['volume'].rolling(window=10).mean() + 1e-8)
    
    # Correlation regime component
    correlation_component = (data['volume_price_corr_3d'] + data['volume_price_corr_10d']) / 2
    
    # Composite factor with regime adaptation
    high_concentration_weight = 0.6
    low_concentration_weight = 0.4
    
    high_regime_factor = (
        efficiency_component * 0.3 +
        level_component * 0.4 +
        volume_momentum_asymmetry * 0.3
    )
    
    low_regime_factor = (
        efficiency_component * 0.4 +
        flow_component * 0.3 +
        correlation_component * 0.3
    )
    
    # Final composite factor
    composite_factor = (
        data['volume_concentration_regime'] * high_regime_factor * high_concentration_weight +
        (1 - data['volume_concentration_regime']) * low_regime_factor * low_concentration_weight
    )
    
    return composite_factor
