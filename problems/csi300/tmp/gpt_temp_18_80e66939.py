import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate daily returns and turnover
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    data['turnover'] = data['volume'] * data['close']
    
    # Multi-Scale Entropy-Regime Framework
    # Price-Volume Entropy Components
    data['price_entropy_5'] = data['close'].diff().abs().rolling(5).apply(
        lambda x: -np.sum((x/x.sum()) * np.log((x/x.sum()).replace(0, 1e-10))) if x.sum() > 0 else 0
    )
    
    data['volume_entropy_10'] = data['volume'].rolling(10).apply(
        lambda x: -np.sum((x/x.sum()) * np.log((x/x.sum()).replace(0, 1e-10)))
    )
    
    data['entropy_divergence'] = (data['volume_entropy_10'] - data['price_entropy_5']) * \
                                np.sign(data['volume_entropy_10'] - data['price_entropy_5'])
    
    # Fractal-Regime Classification
    data['fractal_short'] = (np.log(data['high'].rolling(5).max() - data['low'].rolling(5).min()) / np.log(5)) - \
                           (np.log(data['close'].rolling(5).std()) / np.log(5))
    
    data['fractal_medium'] = (np.log(data['high'].rolling(15).max() - data['low'].rolling(15).min()) / np.log(15)) - \
                            (np.log(data['close'].rolling(15).std()) / np.log(15))
    
    data['fractal_ratio'] = data['fractal_short'] / data['fractal_medium'].replace(0, 1e-10)
    
    # Entropy-Regime Quality
    high_entropy_mask = (data['entropy_divergence'] > 0.1) & (data['fractal_ratio'] > 1.0)
    low_entropy_mask = (data['entropy_divergence'] < -0.1) & (data['fractal_ratio'] < 1.0)
    normal_entropy_mask = ~high_entropy_mask & ~low_entropy_mask
    
    data['entropy_quality'] = np.tanh(data['entropy_divergence'] * data['fractal_ratio'])
    
    # Turnover-Momentum Fractal Patterns
    # Multi-Scale Momentum Components
    data['momentum_3d'] = (data['close'] / data['close'].shift(2) - 1) * \
                         (data['close'].shift(1) / data['close'].shift(3) - 1)
    
    data['momentum_5d_turnover'] = (data['turnover'] / data['turnover'].shift(4) - 1) * \
                                  (data['close'] / data['close'].shift(4) - 1)
    
    data['momentum_consistency'] = ((data['momentum_3d'] > 0) & (data['momentum_5d_turnover'] > 0)).astype(int) + \
                                  ((data['momentum_3d'] < 0) & (data['momentum_5d_turnover'] < 0)).astype(int)
    
    # Turnover Fractal Efficiency
    data['turnover_5d_avg'] = data['turnover'].rolling(5).mean()
    data['turnover_15d_avg'] = data['turnover'].rolling(15).mean()
    data['turnover_ratio'] = (data['turnover_5d_avg'] / data['turnover_15d_avg']) - 1
    
    # Fractal Momentum Score
    data['base_momentum'] = (data['momentum_3d'] + data['momentum_5d_turnover']) / 2
    data['fractal_momentum'] = data['base_momentum'] * data['momentum_consistency'] * data['turnover_ratio']
    
    # Volume-Price Asymmetry with Entropy Confirmation
    # Asymmetry Components
    data['upside_volume'] = np.where(data['close'] > data['close'].shift(1), data['volume'], 0)
    data['upside_volume_ratio'] = data['upside_volume'].rolling(10).mean() / data['volume'].rolling(10).mean()
    
    data['price_up'] = np.maximum(0, data['close'] / data['close'].shift(1) - 1)
    data['price_down'] = np.maximum(0, -(data['close'] / data['close'].shift(1) - 1))
    
    data['price_asymmetry'] = np.log(1 + data['price_up'].rolling(10).sum()) - \
                             np.log(1 + data['price_down'].rolling(10).sum())
    
    data['intraday_pressure'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Entropy-Asymmetry Alignment
    data['volume_price_alignment'] = np.sign(data['price_asymmetry']) * np.sign(data['turnover_ratio'])
    data['alignment_score'] = data['upside_volume_ratio'] * data['price_asymmetry'] * data['volume_price_alignment']
    
    # Asymmetry Confluence
    data['asymmetry_confluence'] = data['alignment_score'] * data['intraday_pressure'] * data['entropy_divergence']
    
    # Volatility-Liquidity Fractal Dynamics
    # Volatility Components
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    data['volatility_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['volume'].replace(0, 1e-10)
    data['volatility_momentum'] = data['true_range'] / data['true_range'].rolling(5).mean() - 1
    
    # Fractal Liquidity
    data['flow_asymmetry'] = abs((data['close'] - data['open']) * data['volume']) / \
                            (abs((data['open'] - data['close'].shift(1)) * data['volume']) + 
                             abs((data['close'] - data['open']) * data['volume']) + 1e-10)
    
    data['liquidity_quality'] = data['volatility_efficiency'] * data['flow_asymmetry'] * data['volatility_momentum']
    
    # Volatility-Liquidity Score
    data['volatility_liquidity_score'] = data['liquidity_quality'] * data['fractal_ratio']
    
    # Turnover Stress with Fractal Persistence
    # Multi-Scale Stress Components
    data['turnover_stress'] = data['turnover'] / data['turnover'].rolling(10).mean()
    
    data['price_turnover_divergence'] = abs(data['fractal_ratio'] - 
                                          (np.log(data['turnover'] / data['turnover'].shift(5)) / np.log(5)))
    
    data['stress_momentum'] = ((data['close'] - data['close'].shift(5)) / data['close'].shift(5)) * \
                             np.log(data['turnover'] / data['turnover'].shift(5))
    
    # Fractal Persistence
    high_stress_mask = data['turnover_stress'] > 1.2
    data['high_stress_persistence'] = np.where(high_stress_mask, data['stress_momentum'], 0)
    data['normal_persistence'] = np.where(~high_stress_mask, data['stress_momentum'], 0)
    
    data['persistence_score'] = (data['high_stress_persistence'] / 
                               data['normal_persistence'].replace(0, 1e-10)) * \
                               (1 - data['price_turnover_divergence'])
    
    # Adaptive Regime Synthesis
    # Regime-Specific Component Integration
    high_entropy_primary = data['entropy_quality'] * data['fractal_momentum']
    high_entropy_secondary = data['asymmetry_confluence'] * data['volatility_liquidity_score']
    high_entropy_alpha = high_entropy_primary * 0.6 + high_entropy_secondary * 0.4
    
    low_entropy_primary = data['fractal_momentum'] * data['persistence_score']
    low_entropy_secondary = data['volatility_liquidity_score'] * data['turnover_ratio']
    low_entropy_alpha = low_entropy_primary * 0.7 + low_entropy_secondary * 0.3
    
    normal_balanced = data['entropy_quality'] * data['fractal_momentum']
    normal_momentum = data['asymmetry_confluence'] * data['persistence_score']
    normal_alpha = normal_balanced * 0.5 + normal_momentum * 0.5
    
    # Stress-Adjusted Integration
    high_entropy_adjustment = high_entropy_alpha * data['turnover_stress']
    low_entropy_adjustment = low_entropy_alpha / data['turnover_stress'].replace(0, 1e-10)
    normal_adjustment = normal_alpha * (1 + abs(data['turnover_stress'] - 1))
    
    # Final Signal
    regime_weighted_alpha = np.where(high_entropy_mask, high_entropy_adjustment,
                                   np.where(low_entropy_mask, low_entropy_adjustment, normal_adjustment))
    
    final_alpha = np.sign(regime_weighted_alpha) * (abs(regime_weighted_alpha) ** (1/3))
    
    return final_alpha
