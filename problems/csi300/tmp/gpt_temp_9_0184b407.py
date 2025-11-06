import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum-Efficiency Synthesis factor
    Combines momentum quality, efficiency convergence, volatility-flow dynamics, and microstructure-liquidity signals
    """
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # 1. Directional Momentum Quality Framework
    # Asymmetric Momentum Calculation
    data['pos_returns_3d'] = data['returns'].rolling(window=3).apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    data['neg_returns_3d'] = data['returns'].rolling(window=3).apply(lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0)
    data['pos_returns_8d'] = data['returns'].rolling(window=8).apply(lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0)
    data['neg_returns_8d'] = data['returns'].rolling(window=8).apply(lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0)
    
    # Momentum asymmetry ratio
    data['momentum_asymmetry'] = np.where(data['neg_returns_8d'] != 0, 
                                         data['pos_returns_8d'] / abs(data['neg_returns_8d']), 
                                         data['pos_returns_8d'])
    
    # Volatility-adjusted momentum persistence
    data['momentum_acceleration'] = data['pos_returns_3d'] - data['pos_returns_8d']
    
    # Flow-enhanced momentum validation
    data['up_volume'] = np.where(data['returns'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['returns'] < 0, data['volume'], 0)
    data['directional_flow'] = (data['up_volume'].rolling(5).sum() - data['down_volume'].rolling(5).sum()) / data['volume'].rolling(5).sum()
    
    # 2. Multi-Horizon Efficiency Convergence
    # Directional Efficiency Analysis
    data['daily_range'] = data['high'] - data['low']
    data['price_change'] = data['close'] - data['open']
    data['efficiency'] = np.where(data['daily_range'] != 0, data['price_change'] / data['daily_range'], 0)
    
    # Up-day and down-day efficiency
    data['up_efficiency'] = np.where(data['returns'] > 0, data['efficiency'], np.nan)
    data['down_efficiency'] = np.where(data['returns'] < 0, data['efficiency'], np.nan)
    
    data['up_eff_rolling'] = data['up_efficiency'].rolling(10, min_periods=5).mean()
    data['down_eff_rolling'] = data['down_efficiency'].rolling(10, min_periods=5).mean()
    data['efficiency_asymmetry'] = np.where(data['down_eff_rolling'] != 0, 
                                          data['up_eff_rolling'] / abs(data['down_eff_rolling']), 
                                          data['up_eff_rolling'])
    
    # Multi-timeframe efficiency alignment
    data['abs_price_change'] = abs(data['close'] - data['close'].shift(1))
    data['range_5d'] = data['high'].rolling(5).max() - data['low'].rolling(5).min()
    data['range_20d'] = data['high'].rolling(20).max() - data['low'].rolling(20).min()
    data['range_60d'] = data['high'].rolling(60).max() - data['low'].rolling(60).min()
    
    data['eff_5d'] = data['abs_price_change'].rolling(5).sum() / data['range_5d']
    data['eff_20d'] = data['abs_price_change'].rolling(20).sum() / data['range_20d']
    data['eff_60d'] = data['abs_price_change'].rolling(60).sum() / data['range_60d']
    
    # Efficiency convergence strength
    data['efficiency_convergence'] = (data['eff_5d'] + data['eff_20d'] + data['eff_60d']) / 3
    
    # 3. Volatility-Flow Regime Integration
    # Asymmetric Volatility Structure
    data['upside_vol'] = data['returns'].rolling(20).apply(lambda x: x[x > 0].std() if len(x[x > 0]) > 1 else 0)
    data['downside_vol'] = data['returns'].rolling(20).apply(lambda x: x[x < 0].std() if len(x[x < 0]) > 1 else 0)
    data['volatility_asymmetry'] = np.where(data['downside_vol'] != 0, 
                                          data['upside_vol'] / data['downside_vol'], 
                                          data['upside_vol'])
    
    # Flow Pressure Acceleration
    data['flow_momentum'] = data['directional_flow'].diff(3)
    
    # 4. Microstructure-Liquidity Convergence
    # Price Discovery Quality Assessment
    data['midpoint'] = (data['high'] + data['low']) / 2
    data['vwap'] = (data['close'] * data['volume']).rolling(5).sum() / data['volume'].rolling(5).sum()
    data['midpoint_deviation'] = (data['midpoint'] - data['vwap']) / data['close']
    
    # Liquidity Provision Asymmetry
    data['upper_range_volume'] = np.where(data['close'] > data['open'], data['volume'], 0)
    data['lower_range_volume'] = np.where(data['close'] < data['open'], data['volume'], 0)
    data['volume_concentration'] = (data['upper_range_volume'].rolling(5).sum() - 
                                  data['lower_range_volume'].rolling(5).sum()) / data['volume'].rolling(5).sum()
    
    # 5. Adaptive Regime Factor Construction
    # Multi-dimensional Signal Integration
    data['momentum_quality'] = (data['momentum_asymmetry'] * data['momentum_acceleration'] * 
                              data['directional_flow']).rolling(5).mean()
    
    data['efficiency_signal'] = (data['efficiency_asymmetry'] * data['efficiency_convergence']).rolling(5).mean()
    
    data['volatility_flow_signal'] = (data['volatility_asymmetry'] * data['flow_momentum']).rolling(5).mean()
    
    data['microstructure_signal'] = (data['midpoint_deviation'] * data['volume_concentration']).rolling(5).mean()
    
    # Final factor construction with regime-adaptive weighting
    data['momentum_efficiency_alignment'] = data['momentum_quality'] * data['efficiency_signal']
    data['volatility_micro_alignment'] = data['volatility_flow_signal'] * data['microstructure_signal']
    
    # Main factor: weighted combination of aligned signals
    data['factor'] = (0.4 * data['momentum_efficiency_alignment'] + 
                     0.3 * data['volatility_micro_alignment'] + 
                     0.3 * (data['momentum_quality'] + data['efficiency_signal'] + 
                           data['volatility_flow_signal'] + data['microstructure_signal']) / 4)
    
    # Acceleration-based enhancement
    data['factor_acceleration'] = data['factor'].diff(3)
    data['final_factor'] = data['factor'] + 0.2 * data['factor_acceleration']
    
    # Clean up and return
    factor_series = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
