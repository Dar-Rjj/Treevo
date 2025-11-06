import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_close2'] = data['close'].shift(2)
    
    # Calculate price differences and ranges
    data['price_diff'] = data['close'] - data['prev_close']
    data['prev_price_diff'] = data['prev_close'] - data['prev_close2']
    data['daily_range'] = data['high'] - data['low']
    data['prev_daily_range'] = data['prev_high'] - data['prev_low']
    data['price_acceleration'] = data['close'] - 2 * data['prev_close'] + data['prev_close2']
    data['vwap'] = data['amount'] / data['volume']
    
    # Momentum Entropy Components
    data['directional_entropy'] = -np.sign(data['price_diff']) * np.log(np.abs(data['price_diff']) / (data['daily_range'] + 1e-8))
    data['velocity_entropy'] = (data['price_diff'] / data['vwap']) * np.log(data['volume'] / (data['prev_volume'] + 1e-8))
    
    # Handle division by zero for acceleration entropy
    prev_price_diff_abs = np.abs(data['prev_price_diff'])
    valid_acceleration = prev_price_diff_abs > 1e-8
    data['acceleration_entropy'] = np.where(
        valid_acceleration,
        (data['price_acceleration'] / data['vwap']) * np.log(np.abs(data['price_diff']) / (prev_price_diff_abs + 1e-8)),
        0
    )
    
    data['momentum_entropy_score'] = data['directional_entropy'] + data['velocity_entropy'] + data['acceleration_entropy']
    
    # Volume-Weighted Entropy
    data['volume_price_entropy'] = data['volume'] * ((data['close'] - data['open']) / (data['daily_range'] + 1e-8)) * np.log(data['volume'] / (data['amount'] / data['close'] + 1e-8))
    
    # Handle division by zero for range-volume entropy
    valid_prev_range = data['prev_daily_range'] > 1e-8
    data['range_volume_entropy'] = np.where(
        valid_prev_range,
        (data['daily_range'] * data['volume'] / data['amount']) * np.log(data['daily_range'] / (data['prev_daily_range'] + 1e-8)),
        0
    )
    
    data['volume_entropy_score'] = data['volume_price_entropy'] + data['range_volume_entropy']
    
    # Entropy Convergence Dynamics
    data['entropy_divergence'] = data['momentum_entropy_score'] - data['volume_entropy_score']
    data['entropy_convergence'] = np.abs(data['momentum_entropy_score']) / (np.abs(data['volume_entropy_score']) + 1e-8)
    data['prev_entropy_divergence'] = data['entropy_divergence'].shift(1)
    data['entropy_persistence'] = np.sign(data['entropy_divergence']) * np.sign(data['prev_entropy_divergence'])
    
    # Regime Detection
    data['high_entropy_regime'] = (data['momentum_entropy_score'] > data['volume_entropy_score']) & (data['entropy_divergence'] > 0)
    data['low_entropy_regime'] = (data['momentum_entropy_score'] < data['volume_entropy_score']) & (data['entropy_divergence'] < 0)
    data['balanced_entropy_regime'] = ~(data['high_entropy_regime'] | data['low_entropy_regime'])
    
    data['volume_dominant_regime'] = (data['volume_entropy_score'] > np.abs(data['momentum_entropy_score'])) & (data['volume'] > data['prev_volume'])
    data['price_dominant_regime'] = (data['momentum_entropy_score'] > np.abs(data['volume_entropy_score'])) & (np.abs(data['price_diff']) > (data['prev_daily_range'] / 2))
    data['mixed_regime'] = ~(data['volume_dominant_regime'] | data['price_dominant_regime'])
    
    # Regime Persistence Calculation
    data['same_entropy_regime'] = (data['high_entropy_regime'] == data['high_entropy_regime'].shift(1)) & (data['low_entropy_regime'] == data['low_entropy_regime'].shift(1))
    data['same_volume_regime'] = (data['volume_dominant_regime'] == data['volume_dominant_regime'].shift(1)) & (data['price_dominant_regime'] == data['price_dominant_regime'].shift(1))
    
    regime_persistence = []
    for i in range(len(data)):
        if i < 2:
            regime_persistence.append(0)
        else:
            count = 0
            if data['same_entropy_regime'].iloc[i] and data['same_entropy_regime'].iloc[i-1]:
                count += 1
            if data['same_volume_regime'].iloc[i] and data['same_volume_regime'].iloc[i-1]:
                count += 1
            regime_persistence.append(count)
    
    data['regime_persistence'] = regime_persistence
    
    # Regime Volatility and Momentum
    data['regime_volatility'] = (data['daily_range'] / (data['prev_daily_range'] + 1e-8)) * data['regime_persistence']
    data['regime_momentum'] = (data['price_diff'] / data['vwap']) * data['regime_persistence']
    
    # Cross-Scale Entropy Integration
    # Ultra-Short Scale (1-2 days)
    data['gap_entropy'] = ((data['close'] - data['open']) / (data['prev_daily_range'] + 1e-8)) * np.log(np.abs(data['close'] - data['open']) / (data['daily_range'] + 1e-8))
    data['volume_spike_entropy'] = (data['volume'] / (data['prev_volume'] + 1e-8)) * np.log(data['volume'] / (data['prev_volume'] + 1e-8))
    data['ultra_short_entropy_factor'] = data['gap_entropy'] * data['volume_spike_entropy'] * data['entropy_convergence']
    
    # Short-Term Scale (3-5 days)
    # Calculate rolling momentum persistence
    momentum_signs = []
    for i in range(len(data)):
        if i < 4:
            momentum_signs.append(0)
        else:
            window = data.iloc[i-4:i+1]
            positive_count = (window['price_diff'] > 0).sum()
            volume_ratio = window['volume'].iloc[-1] / (window['volume'].iloc[0] + 1e-8)
            momentum_signs.append(positive_count * np.log(volume_ratio + 1e-8))
    
    data['momentum_persistence_entropy'] = momentum_signs
    
    # Calculate range expansion
    range_expansion = []
    for i in range(len(data)):
        if i < 4:
            range_expansion.append(0)
        else:
            window = data.iloc[i-4:i+1]
            max_high = window['high'].max()
            min_low = window['low'].min()
            base_range = data['prev_daily_range'].iloc[i-4]
            if base_range > 1e-8:
                expansion = (max_high - min_low) / base_range
                range_expansion.append(expansion * np.log(expansion + 1e-8))
            else:
                range_expansion.append(0)
    
    data['range_expansion_entropy'] = range_expansion
    data['short_term_entropy_factor'] = data['momentum_persistence_entropy'] * data['range_expansion_entropy'] * data['entropy_divergence']
    
    # Medium-Term Scale (6-10 days)
    # Calculate volume trend entropy
    volume_trend = []
    for i in range(len(data)):
        if i < 6:
            volume_trend.append(0)
        else:
            volume_ratio = data['volume'].iloc[i] / (data['volume'].iloc[i-6] + 1e-8)
            volume_trend.append(volume_ratio * np.log(volume_ratio + 1e-8))
    
    data['volume_trend_entropy'] = volume_trend
    
    # Calculate price trend entropy
    price_trend = []
    for i in range(len(data)):
        if i < 6:
            price_trend.append(0)
        else:
            window = data.iloc[i-6:i+1]
            max_high = window['high'].max()
            min_low = window['low'].min()
            price_range = max_high - min_low
            price_change = data['close'].iloc[i] - data['close'].iloc[i-6]
            
            if price_range > 1e-8:
                trend = price_change / price_range
                price_trend.append(trend * np.log(np.abs(price_change) / (price_range + 1e-8)))
            else:
                price_trend.append(0)
    
    data['price_trend_entropy'] = price_trend
    data['medium_term_entropy_factor'] = data['volume_trend_entropy'] * data['price_trend_entropy'] * data['entropy_persistence']
    
    # Regime-Adaptive Factor Construction
    # Entropy-Regime Factors
    data['high_entropy_factor'] = data['momentum_entropy_score'] * data['entropy_divergence'] * data['regime_volatility']
    data['low_entropy_factor'] = data['volume_entropy_score'] * data['entropy_convergence'] * data['regime_momentum']
    data['balanced_entropy_factor'] = (data['momentum_entropy_score'] + data['volume_entropy_score']) * data['entropy_persistence']
    
    # Volume-Price Regime Factors
    data['volume_dominant_factor'] = data['volume_entropy_score'] * (data['volume'] / (data['prev_volume'] + 1e-8)) * data['regime_persistence']
    data['price_dominant_factor'] = data['momentum_entropy_score'] * (data['price_diff'] / (data['daily_range'] + 1e-8)) * data['regime_volatility']
    data['mixed_regime_factor'] = (data['volume_entropy_score'] + data['momentum_entropy_score']) * data['regime_momentum']
    
    # Cross-Regime Integration
    data['entropy_regime_core'] = np.where(
        data['high_entropy_regime'],
        data['high_entropy_factor'],
        np.where(
            data['low_entropy_regime'],
            data['low_entropy_factor'],
            data['balanced_entropy_factor']
        )
    )
    
    data['volume_price_core'] = np.where(
        data['volume_dominant_regime'],
        data['volume_dominant_factor'],
        np.where(
            data['price_dominant_regime'],
            data['price_dominant_factor'],
            data['mixed_regime_factor']
        )
    )
    
    data['regime_switching_core'] = data['entropy_regime_core'] * data['volume_price_core'] * data['regime_persistence']
    
    # Multi-Scale Entropy Enhancement
    data['ultra_short_weight'] = 1 / (1 + np.abs(data['ultra_short_entropy_factor']))
    data['short_term_weight'] = 1 / (1 + np.abs(data['short_term_entropy_factor']))
    data['medium_term_weight'] = 1 / (1 + np.abs(data['medium_term_entropy_factor']))
    
    data['ultra_short_enhanced'] = data['regime_switching_core'] * data['ultra_short_entropy_factor'] * data['ultra_short_weight']
    data['short_term_enhanced'] = data['regime_switching_core'] * data['short_term_entropy_factor'] * data['short_term_weight']
    data['medium_term_enhanced'] = data['regime_switching_core'] * data['medium_term_entropy_factor'] * data['medium_term_weight']
    
    data['weighted_scale_sum'] = data['ultra_short_enhanced'] + data['short_term_enhanced'] + data['medium_term_enhanced']
    data['multi_scale_entropy_alpha'] = data['weighted_scale_sum'] * (1 / (1 + np.abs(data['entropy_divergence'])))
    
    # Dynamic Risk-Entropy Adjustment
    data['price_risk_entropy'] = (data['daily_range'] / (data['prev_daily_range'] + 1e-8)) * np.log(data['daily_range'] / (data['prev_daily_range'] + 1e-8))
    data['volume_risk_entropy'] = (data['volume'] / (data['prev_volume'] + 1e-8)) * np.log(data['volume'] / (data['prev_volume'] + 1e-8))
    data['combined_risk_entropy'] = data['price_risk_entropy'] * data['volume_risk_entropy']
    
    data['high_risk_regime'] = (data['combined_risk_entropy'] > 0) & (data['price_risk_entropy'] > data['volume_risk_entropy'])
    data['low_risk_regime'] = (data['combined_risk_entropy'] < 0) & (data['volume_risk_entropy'] > data['price_risk_entropy'])
    data['moderate_risk_regime'] = ~(data['high_risk_regime'] | data['low_risk_regime'])
    
    data['high_risk_scaling'] = 1 / (1 + np.abs(data['combined_risk_entropy']))
    data['low_risk_scaling'] = 1 / (1 + np.abs(data['price_risk_entropy'] - data['volume_risk_entropy']))
    data['moderate_risk_scaling'] = (data['high_risk_scaling'] + data['low_risk_scaling']) / 2
    
    data['adaptive_risk_weight'] = np.where(
        data['high_risk_regime'],
        data['high_risk_scaling'],
        np.where(
            data['low_risk_regime'],
            data['low_risk_scaling'],
            data['moderate_risk_scaling']
        )
    )
    
    # Final Alpha Construction
    data['entropy_momentum_core'] = data['momentum_entropy_score'] * data['regime_switching_core']
    data['volume_entropy_core'] = data['volume_entropy_score'] * data['multi_scale_entropy_alpha']
    data['risk_entropy_core'] = data['combined_risk_entropy'] * data['adaptive_risk_weight']
    
    data['primary_alpha'] = data['entropy_momentum_core'] * data['volume_entropy_core']
    data['risk_adjusted_alpha'] = data['primary_alpha'] * data['risk_entropy_core']
    data['regime_validated_alpha'] = data['risk_adjusted_alpha'] * data['regime_persistence']
    
    # Final Output Factor
    data['momentum_volume_entropy_alpha'] = data['regime_validated_alpha'] * data['entropy_convergence']
    data['final_regime_switching_alpha'] = data['momentum_volume_entropy_alpha'] * (1 / (1 + np.abs(data['entropy_divergence'])))
    
    # Return the final factor
    return data['final_regime_switching_alpha']
