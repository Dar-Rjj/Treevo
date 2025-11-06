import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Multi-Scale Momentum Components
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(4) - 1
    
    # Turnover Momentum Components
    data['turnover_3d'] = (
        data['volume'] * data['close'] + 
        data['volume'].shift(1) * data['close'].shift(1) + 
        data['volume'].shift(2) * data['close'].shift(2)
    ) / 3
    data['turnover_5d'] = (
        data['volume'] * data['close'] + 
        data['volume'].shift(1) * data['close'].shift(1) + 
        data['volume'].shift(2) * data['close'].shift(2) + 
        data['volume'].shift(3) * data['close'].shift(3) + 
        data['volume'].shift(4) * data['close'].shift(4)
    ) / 5
    data['turnover_momentum'] = data['turnover_3d'] / data['turnover_5d'] - 1
    
    # Fractal Momentum-Turnover Divergence
    data['momentum_decay_ratio'] = (data['momentum_1d'] / data['momentum_3d']) * (data['volume'] / data['volume'].shift(1))
    data['multi_scale_divergence'] = (data['momentum_5d'] - data['momentum_3d']) * data['turnover_momentum']
    data['fractal_divergence_coherence'] = data['momentum_decay_ratio'] * data['multi_scale_divergence']
    
    # Volatility Asymmetry Components
    data['upside_vol'] = (data['high'] - data['open']) / data['open']
    data['downside_vol'] = (data['open'] - data['low']) / data['open']
    data['fractal_asymmetry'] = (data['upside_vol'] / data['downside_vol']) / (data['upside_vol'].shift(1) / data['downside_vol'].shift(1))
    
    # Price Rejection Components
    data['upper_shadow_rejection'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['lower_shadow_rejection'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['rejection_efficiency'] = np.abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) / data['amount']
    
    # Asymmetry-Rejection Synthesis
    data['volatility_rejection_factor'] = data['fractal_asymmetry'] * data['rejection_efficiency']
    data['shadow_asymmetry'] = data['upper_shadow_rejection'] - data['lower_shadow_rejection']
    data['asymmetry_flow'] = data['volatility_rejection_factor'] * data['shadow_asymmetry'] * (data['volume'] / data['volume'].shift(1))
    
    # Volume Confirmation Patterns
    returns = data['close'].pct_change()
    up_days_mask = returns > 0
    data['up_day_volume_ratio'] = (
        data['volume'].rolling(window=5).apply(lambda x: x[up_days_mask.loc[x.index]].mean() if up_days_mask.loc[x.index].any() else 0) / 
        data['volume'].rolling(window=5).mean()
    )
    
    positive_returns_sum = returns.rolling(window=5).apply(lambda x: x[x > 0].sum())
    negative_returns_sum = returns.rolling(window=5).apply(lambda x: np.abs(x[x < 0]).sum())
    data['price_movement_asymmetry'] = np.log1p(positive_returns_sum) - np.log1p(negative_returns_sum)
    data['volume_price_confirmation'] = data['up_day_volume_ratio'] * data['price_movement_asymmetry']
    
    # Flow Compression Components
    data['flow_imbalance'] = (data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    data['range_compression'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['compression_efficiency'] = np.abs(data['flow_imbalance']) / data['range_compression']
    
    # Flow Dynamics Synthesis
    data['confirmed_flow'] = data['volume_price_confirmation'] * data['flow_imbalance']
    data['compressed_momentum'] = data['compression_efficiency'] * (data['close'] / data['close'].shift(1) - 1)
    data['flow_coherence'] = data['confirmed_flow'] * data['compressed_momentum']
    
    # Volatility-Regime Adaptive Factors
    data['volatility_component'] = data['fractal_divergence_coherence'] * data['asymmetry_flow']
    data['efficiency_adjustment'] = data['rejection_efficiency'] * (data['volume'] / data['volume'].shift(1))
    data['high_volatility_factor'] = data['volatility_component'] * data['efficiency_adjustment']
    
    data['compression_component'] = data['compression_efficiency'] * np.abs(data['rejection_efficiency'] - 1)
    data['flow_confirmation'] = data['flow_coherence'] * data['volume_price_confirmation']
    data['low_volatility_factor'] = data['compression_component'] * data['flow_confirmation']
    
    # Transition Detection
    data['price_flow_divergence'] = np.abs((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) - np.abs(data['flow_imbalance'])
    data['volatility_efficiency_gap'] = data['fractal_asymmetry'] - data['compression_efficiency']
    data['transition_factor'] = data['price_flow_divergence'] * data['volatility_efficiency_gap'] * (data['volume'] / data['volume'].shift(1))
    
    # Final Multi-Regime Synthesis
    data['base_divergence'] = data['fractal_divergence_coherence'] * data['asymmetry_flow']
    data['volume_adjustment'] = data['volume_price_confirmation'] * (data['volume'] / data['volume'].shift(1))
    data['core_factor'] = data['base_divergence'] * data['volume_adjustment']
    
    data['high_volatility_weighting'] = data['high_volatility_factor'] * np.abs(data['flow_imbalance'])
    data['low_volatility_weighting'] = data['low_volatility_factor'] * data['compression_efficiency']
    data['transition_weighting'] = data['transition_factor'] * np.abs(data['price_flow_divergence'])
    
    data['regime_adaptive_blend'] = data['core_factor'] * (data['high_volatility_weighting'] + data['low_volatility_weighting'] + data['transition_weighting'])
    data['flow_consistency_check'] = data['flow_coherence'] * (data['volume'] / data['volume'].shift(1))
    
    # Final alpha factor
    alpha = data['regime_adaptive_blend'] * data['flow_consistency_check']
    
    return alpha
