import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Gap Momentum with Fractal Liquidity Acceleration
    """
    data = df.copy()
    
    # Multi-Scale Gap Momentum Analysis
    # Overnight Gap Momentum with Residual Adjustment
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['market_overnight_gap'] = data['overnight_gap'].rolling(window=5, min_periods=3).mean()
    data['adj_overnight_gap'] = data['overnight_gap'] - data['market_overnight_gap']
    data['overnight_momentum_5d'] = data['adj_overnight_gap'].rolling(window=5, min_periods=3).sum()
    
    # Intraday Gap Momentum with Fractal Quality
    data['intraday_gap'] = (data['close'] - data['open']) / data['open']
    data['market_intraday_gap'] = data['intraday_gap'].rolling(window=5, min_periods=3).mean()
    data['adj_intraday_gap'] = data['intraday_gap'] - data['market_intraday_gap']
    data['intraday_momentum_3d'] = data['adj_intraday_gap'].rolling(window=3, min_periods=2).sum()
    data['intraday_momentum_5d'] = data['adj_intraday_gap'].rolling(window=5, min_periods=3).sum()
    
    # Gap Divergence Analysis with Momentum Quality
    data['gap_divergence'] = np.sign(data['overnight_gap']) != np.sign(data['intraday_gap'])
    data['divergence_strength'] = np.abs(data['overnight_gap']) * np.abs(data['intraday_gap'])
    data['momentum_quality'] = np.sign(data['overnight_gap']) * np.sign(data['intraday_gap']) * data['divergence_strength']
    
    # Fractal Volume Acceleration with Microstructure Patterns
    # Multi-Scale Volume Acceleration
    data['volume_accel_short'] = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8)
    data['volume_accel_medium'] = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    data['volume_accel_divergence'] = data['volume_accel_short'] - data['volume_accel_medium']
    
    # Volume-Price Fractal Efficiency
    data['directional_volume_efficiency'] = (data['close'] - data['open']) * data['volume'] / (data['high'] - data['low'] + 1e-8)
    data['volume_persistence_5d'] = data['volume'].rolling(window=5, min_periods=3).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    data['volume_persistence_20d'] = data['volume'].rolling(window=20, min_periods=10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    # Order Flow Dynamics with Gap Momentum
    data['signed_volume_gap'] = np.sign(data['close'] - data['open']) * data['volume']
    data['order_flow_3d'] = data['signed_volume_gap'].rolling(window=3, min_periods=2).sum()
    data['order_flow_persistence'] = data['signed_volume_gap'].rolling(window=10, min_periods=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    # Dynamic Volatility-Regime Gap Processing
    # Multi-Scale Volatility Environment
    data['true_range'] = np.maximum(data['high'] - data['low'], 
                                   np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                                             np.abs(data['low'] - data['close'].shift(1))))
    data['gap_volatility_5d'] = data['overnight_gap'].rolling(window=5, min_periods=3).std()
    data['range_volatility_20d'] = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean()
    data['volatility_ratio'] = data['gap_volatility_5d'] / (data['range_volatility_20d'] + 1e-8)
    
    # Volatility-Weighted Gap Components
    data['daily_range'] = data['high'] - data['low']
    data['range_volatility_10d'] = data['daily_range'].rolling(window=10, min_periods=5).std()
    data['volatility_weight'] = 1 / (data['range_volatility_10d'] + 1e-8)
    
    # Liquidity Gradient Gap Enhancement
    data['volume_price_impact'] = (data['close'] - data['open']) / (data['volume'] + 1e-8)
    data['liquidity_gradient'] = data['volume_price_impact'].rolling(window=5, min_periods=3).mean()
    
    # Volume Confirmation for Gap Signals
    data['volume_spike'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_confirmation'] = data['volume_spike'] * np.sign(data['overnight_gap']) * np.sign(data['intraday_gap'])
    
    # Composite Factor Construction
    # Regime-Adaptive Gap Momentum
    high_vol_regime = data['volatility_ratio'] > data['volatility_ratio'].rolling(window=20, min_periods=10).quantile(0.7)
    low_vol_regime = data['volatility_ratio'] < data['volatility_ratio'].rolling(window=20, min_periods=10).quantile(0.3)
    
    # High volatility regime processing
    high_vol_component = (data['gap_divergence'] * data['divergence_strength'] * 
                         data['volume_accel_divergence'] * data['volatility_weight'])
    
    # Low volatility regime processing
    low_vol_component = (data['overnight_momentum_5d'] * data['intraday_momentum_5d'] * 
                        data['volume_confirmation'] * (1 + data['volume_accel_medium']))
    
    # Microstructure-Enhanced Gap Momentum
    microstructure_component = (data['order_flow_3d'] * data['directional_volume_efficiency'] * 
                              data['liquidity_gradient'] * data['momentum_quality'])
    
    # Quality-Weighted Final Factor
    regime_component = np.where(high_vol_regime, high_vol_component,
                               np.where(low_vol_regime, low_vol_component,
                                       (high_vol_component + low_vol_component) / 2))
    
    # Final composite factor
    final_factor = (regime_component * 0.4 + 
                   microstructure_component * 0.3 + 
                   data['volume_persistence_5d'] * data['order_flow_persistence'] * 0.3)
    
    return pd.Series(final_factor, index=data.index)
