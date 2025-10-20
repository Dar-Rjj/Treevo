import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency Momentum Framework
    # Price Efficiency Calculation
    for window in [5, 10, 21]:
        # Calculate price efficiency: (Close_t - Close_{t-window}) / (High_{t-window+1:t} - Low_{t-window+1:t})
        price_change = data['close'] - data['close'].shift(window)
        range_window = (data['high'].rolling(window=window).max() - 
                       data['low'].rolling(window=window).min())
        data[f'efficiency_{window}d'] = price_change / (range_window + 1e-8)
    
    # Efficiency Momentum Generation
    data['efficiency_momentum_short'] = data['efficiency_5d'] - data['efficiency_10d']
    data['efficiency_momentum_medium'] = data['efficiency_10d'] - data['efficiency_21d']
    data['efficiency_convergence'] = (np.sign(data['efficiency_momentum_short']) * 
                                     np.sign(data['efficiency_momentum_medium']))
    
    # Efficiency Divergence Patterns
    efficiencies = [data['efficiency_5d'], data['efficiency_10d'], data['efficiency_21d']]
    data['efficiency_divergence_max'] = pd.concat(efficiencies, axis=1).diff(axis=1).abs().max(axis=1)
    data['efficiency_variance'] = pd.concat(efficiencies, axis=1).var(axis=1)
    
    # Nonlinear Volume-Adapted Range Analysis
    # Volume-Adjusted Range Calculation
    data['daily_range'] = data['high'] - data['low']
    data['volume_adjusted_range'] = data['daily_range'] * np.log(data['volume'] + 1)
    data['range_efficiency_ratio'] = (data['close'] - data['close'].shift(1)) / (data['volume_adjusted_range'] + 1e-8)
    
    # Volume Persistence Patterns
    data['volume_autocorr_5d'] = data['volume'].rolling(window=5).apply(
        lambda x: x.corr(pd.Series(x).shift(1).fillna(method='bfill')), raw=False)
    data['volume_trend_strength'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['volume_breakout'] = (data['volume'] > 1.5 * data['volume'].rolling(window=20).mean()).astype(float)
    
    # Nonlinear Volume Confirmation
    data['volume_range_momentum'] = data['volume_adjusted_range'].diff(5)
    data['range_efficiency_divergence'] = data['efficiency_5d'] - data['range_efficiency_ratio']
    data['volume_persistence_confirmation'] = data['volume_autocorr_5d'] * data['efficiency_momentum_short']
    
    # Price-Volume Synchronization with Structural Breaks
    # Multi-dimensional Efficiency Correlations
    data['price_volume_corr_10d'] = data['close'].rolling(window=10).corr(data['volume'])
    data['price_range_corr_10d'] = data['close'].rolling(window=10).corr(data['daily_range'])
    data['volume_range_efficiency_corr'] = data['volume_adjusted_range'].rolling(window=10).corr(data['efficiency_5d'])
    
    # Structural Break Detection
    data['efficiency_variance_ratio'] = (data['efficiency_5d'].rolling(window=5).var() / 
                                        (data['efficiency_5d'].rolling(window=20).var() + 1e-8))
    data['volume_relationship_shift'] = data['price_volume_corr_10d'].diff(5)
    data['regime_change'] = (data['efficiency_variance_ratio'] > 2.0).astype(float)
    
    # Synchronization Strength Assessment
    data['correlation_vs_history'] = (data['price_volume_corr_10d'] - 
                                     data['price_volume_corr_10d'].rolling(window=20).mean())
    data['multi_dim_alignment'] = (data['price_volume_corr_10d'] + 
                                  data['price_range_corr_10d'] + 
                                  data['volume_range_efficiency_corr']) / 3
    data['break_enhanced_sync'] = data['multi_dim_alignment'] * data['regime_change']
    
    # Volatility-Regime Adaptive Integration
    # Efficiency Regime Classification
    data['high_efficiency'] = (data['efficiency_5d'] > data['efficiency_5d'].rolling(window=20).mean()).astype(float)
    data['low_efficiency'] = (data['efficiency_5d'] < data['efficiency_5d'].rolling(window=20).mean()).astype(float)
    data['efficiency_transition'] = data['high_efficiency'].diff().abs()
    
    # Volatility Assessment
    data['range_regime'] = data['daily_range'] / (data['daily_range'].rolling(window=20).mean() + 1e-8)
    data['volume_adjusted_vol'] = (data['volume_adjusted_range'] / 
                                  (data['volume_adjusted_range'].rolling(window=20).mean() + 1e-8))
    data['volatility_persistence'] = data['daily_range'].rolling(window=5).apply(
        lambda x: x.corr(pd.Series(x).shift(1).fillna(method='bfill')), raw=False)
    
    # Adaptive Signal Weighting
    data['weight_high_eff_high_vol'] = ((data['high_efficiency'] == 1) & 
                                       (data['volume_adjusted_vol'] > 1)).astype(float) * 1.5
    data['weight_low_eff_break'] = ((data['low_efficiency'] == 1) & 
                                   (data['regime_change'] == 1)).astype(float) * 1.3
    data['weight_volume_persistence'] = ((data['volume_persistence_confirmation'] > 0) & 
                                        (data['efficiency_convergence'] > 0)).astype(float) * 1.2
    data['weight_divergence_caution'] = ((data['efficiency_divergence_max'] > 
                                         data['efficiency_divergence_max'].rolling(window=20).mean()) & 
                                        (data['volatility_persistence'] < 0)).astype(float) * 0.6
    
    # Nonlinear Acceleration Patterns
    # Efficiency Acceleration Metrics
    data['efficiency_acceleration'] = (data['efficiency_momentum_short'] - 
                                      data['efficiency_momentum_medium'])
    data['acceleration_roc'] = data['efficiency_acceleration'].diff()
    data['acceleration_persistence'] = (data['efficiency_acceleration'] > 0).rolling(window=5).sum()
    
    # Volume-Confirmed Acceleration
    data['volume_trend_alignment'] = (np.sign(data['volume_trend_strength'] - 1) * 
                                     np.sign(data['efficiency_acceleration']))
    data['volume_range_acceleration'] = data['volume_adjusted_range'].diff(5)
    data['nonlinear_breakout_acceleration'] = data['volume_breakout'] * data['efficiency_acceleration']
    
    # Structural Break Enhanced Acceleration
    data['regime_enhanced_acceleration'] = data['regime_change'] * data['efficiency_acceleration'].abs()
    data['volume_persistence_acceleration'] = data['volume_autocorr_5d'] * data['acceleration_persistence']
    
    # Composite Alpha Generation
    # Base efficiency momentum with volume weighting
    base_signal = (data['efficiency_momentum_short'] * 
                  (1 + data['volume_trend_strength'] - 1))
    
    # Volume-adjusted range efficiency signals
    range_signal = data['range_efficiency_ratio'] * data['volume_adjusted_range']
    
    # Structural break detection enhancement
    break_signal = data['break_enhanced_sync'] * data['regime_enhanced_acceleration']
    
    # Volatility-regime adaptive filtering
    volatility_filter = (data['weight_high_eff_high_vol'] + 
                        data['weight_low_eff_break'] + 
                        data['weight_volume_persistence'] + 
                        data['weight_divergence_caution'])
    
    # Acceleration persistence multiplier
    acceleration_multiplier = 1 + (data['acceleration_persistence'] * 0.1)
    
    # Multi-dimensional synchronization confirmation
    sync_confirmation = (data['multi_dim_alignment'] * 
                        data['volume_persistence_acceleration'])
    
    # Final composite alpha
    alpha = (base_signal + range_signal + break_signal) * volatility_filter * acceleration_multiplier * sync_confirmation
    
    return alpha
