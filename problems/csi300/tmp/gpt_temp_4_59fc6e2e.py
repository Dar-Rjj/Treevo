import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Momentum Assessment
    data['short_price_momentum'] = (data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 1e-8)
    data['short_volume_momentum'] = (data['volume'] - data['volume'].shift(3)) / (data['volume'].shift(3) + 1e-8)
    data['medium_price_momentum'] = (data['close'] - data['close'].shift(8)) / (data['close'].shift(8) + 1e-8)
    data['medium_volume_momentum'] = (data['volume'] - data['volume'].shift(8)) / (data['volume'].shift(8) + 1e-8)
    
    data['momentum_ratio'] = data['short_price_momentum'] / (data['medium_price_momentum'] + 1e-8)
    data['momentum_regime'] = np.sign(data['momentum_ratio'] - 1)
    
    # Range Efficiency Patterns
    data['daily_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['short_range_efficiency'] = data['daily_efficiency'].rolling(window=5, min_periods=1).mean()
    data['medium_range_efficiency'] = data['daily_efficiency'].rolling(window=10, min_periods=1).mean()
    data['efficiency_divergence'] = data['short_range_efficiency'] - data['medium_range_efficiency']
    
    data['efficiency_momentum'] = data['short_range_efficiency'] - data['daily_efficiency'].shift(1).rolling(window=5, min_periods=1).mean()
    data['efficiency_acceleration'] = data['efficiency_momentum'] / (data['short_range_efficiency'] + 0.001)
    
    data['price_range_corr'] = data['short_price_momentum'].rolling(window=5, min_periods=1).corr(data['short_range_efficiency'])
    data['range_momentum_regime'] = np.sign(data['price_range_corr']) * data['momentum_regime']
    
    # Price-Volume Impact Dynamics
    data['price_change'] = data['close'].diff()
    
    # Short-term Impact (3-day)
    data['short_impact_numerator'] = (abs(data['price_change']) * data['volume']).rolling(window=3, min_periods=1).sum()
    data['short_impact_denom'] = data['volume'].rolling(window=3, min_periods=1).sum()
    data['short_impact'] = data['short_impact_numerator'] / (data['short_impact_denom'] + 1e-8)
    
    # Medium-term Impact (8-day)
    data['medium_impact_numerator'] = (abs(data['price_change']) * data['volume']).rolling(window=8, min_periods=1).sum()
    data['medium_impact_denom'] = data['volume'].rolling(window=8, min_periods=1).sum()
    data['medium_impact'] = data['medium_impact_numerator'] / (data['medium_impact_denom'] + 1e-8)
    
    data['impact_scaling_ratio'] = data['short_impact'] / (data['medium_impact'] + 1e-8)
    
    # Range Impact Dimension
    data['range_impact_dim'] = (abs(data['price_change']) * data['volume'] / (data['high'] - data['low'] + 1e-8)).rolling(window=5, min_periods=1).sum()
    data['range_impact_corr'] = data['short_range_efficiency'].rolling(window=10, min_periods=1).corr(data['range_impact_dim'])
    
    data['impact_consistency'] = np.sign(data['impact_scaling_ratio'] - 1) * np.sign(data['range_impact_corr'])
    
    # Range Multi-impact Spectrum
    data['daily_range'] = data['high'] - data['low']
    avg_range = data['daily_range'].rolling(window=10, min_periods=1).mean()
    
    high_range_mask = data['daily_range'] > avg_range
    low_range_mask = data['daily_range'] <= avg_range
    
    def calc_weighted_impact(data, mask, window=10):
        impact_values = []
        for i in range(len(data)):
            if i < window-1:
                impact_values.append(np.nan)
                continue
            window_data = data.iloc[i-window+1:i+1]
            window_mask = mask.iloc[i-window+1:i+1]
            if window_mask.sum() == 0:
                impact_values.append(np.nan)
                continue
            numerator = (abs(window_data['price_change']) * window_data['volume'])[window_mask].sum()
            denominator = window_data['volume'][window_mask].sum()
            impact_values.append(numerator / (denominator + 1e-8))
        return pd.Series(impact_values, index=data.index)
    
    data['high_range_impact'] = calc_weighted_impact(data, high_range_mask)
    data['low_range_impact'] = calc_weighted_impact(data, low_range_mask)
    data['range_multiimpact_width'] = data['high_range_impact'] - data['low_range_impact']
    
    # Volume Range Patterns
    data['volume_change'] = data['volume'].diff()
    data['volume_range_impact'] = ((abs(data['volume_change']) * data['daily_range']).rolling(window=5, min_periods=1).sum() / 
                                  (data['daily_range'].rolling(window=5, min_periods=1).sum() + 1e-8))
    data['volume_range_efficiency'] = data['short_range_efficiency'] * data['volume_range_impact']
    
    data['range_volume_impact_corr'] = data['range_multiimpact_width'].rolling(window=10, min_periods=1).corr(data['volume_range_impact'])
    data['range_impact_regime'] = np.sign(data['range_volume_impact_corr']) * np.sign(data['range_multiimpact_width'])
    
    # Impact Efficiency Metrics
    data['short_range_efficiency_impact'] = abs(data['close'] - data['close'].shift(3)) / (
        (abs(data['price_change']) * data['volume'] / (data['high'] - data['low'] + 1e-8)).rolling(window=3, min_periods=1).sum() + 1e-8)
    
    data['medium_range_efficiency_impact'] = abs(data['close'] - data['close'].shift(8)) / (
        (abs(data['price_change']) * data['volume'] / (data['high'] - data['low'] + 1e-8)).rolling(window=8, min_periods=1).sum() + 1e-8)
    
    data['range_efficiency_impact_ratio'] = data['short_range_efficiency_impact'] / (data['medium_range_efficiency_impact'] + 1e-8)
    
    data['volume_range_movement'] = ((abs(data['price_change']) / (data['high'] - data['low'] + 1e-8)).rolling(window=5, min_periods=1).sum() / 
                                    (abs(data['volume_change']) * data['daily_range']).rolling(window=5, min_periods=1).sum() + 1e-8)
    
    data['volume_range_efficiency_impact'] = data['volume_range_movement'] * data['impact_scaling_ratio']
    
    data['range_volume_impact_alignment'] = data['range_efficiency_impact_ratio'] * data['volume_range_efficiency_impact']
    data['range_impact_efficiency_score'] = data['range_volume_impact_alignment'] * data['impact_consistency']
    
    # Regime-Adaptive Signal Construction
    momentum_breakout = abs(data['momentum_ratio'])
    
    # High Momentum-Range Regime
    high_regime_mask = (momentum_breakout > 1.5) | (data['efficiency_divergence'] > 0.1)
    data['range_impact_momentum'] = data['impact_scaling_ratio'] * data['range_multiimpact_width']
    data['momentum_range_impact_alignment'] = data['momentum_regime'] * data['range_impact_corr']
    data['high_momentum_range_signal'] = data['range_impact_momentum'] * data['momentum_range_impact_alignment'] * data['range_impact_efficiency_score']
    
    # Low Momentum-Range Regime
    low_regime_mask = (momentum_breakout <= 1.5) & (data['efficiency_divergence'] <= 0.1)
    data['range_multiimpact_convergence'] = data['range_multiimpact_width'] * data['volume_range_impact']
    data['range_impact_regime_alignment'] = data['impact_consistency'] * data['range_impact_regime']
    data['low_momentum_range_signal'] = data['range_multiimpact_convergence'] * data['range_impact_regime_alignment'] * data['range_impact_corr']
    
    # Transition Regime
    transition_mask = (abs(momentum_breakout - 1) < 0.2) & (abs(data['efficiency_divergence']) < 0.05)
    data['range_momentum_signal'] = data['efficiency_acceleration'] * data['range_momentum_regime']
    data['range_impact_transition_signal'] = data['impact_scaling_ratio'] * data['range_impact_corr']
    data['range_multiimpact_transition'] = data['range_impact_regime'] * data['range_impact_efficiency_score']
    data['transition_signal'] = data['range_momentum_signal'] * data['range_impact_transition_signal'] * data['range_multiimpact_transition']
    
    # Cross-Regime Signal Integration
    data['high_momentum_confidence'] = abs(momentum_breakout - 1) * abs(data['efficiency_divergence'])
    data['low_momentum_confidence'] = 1 / (abs(momentum_breakout - 1) * abs(data['efficiency_divergence']) + 0.001)
    data['transition_confidence'] = 1 - np.maximum(data['high_momentum_confidence'], data['low_momentum_confidence'])
    
    data['high_weighted'] = data['high_momentum_range_signal'] * data['high_momentum_confidence']
    data['low_weighted'] = data['low_momentum_range_signal'] * data['low_momentum_confidence']
    data['transition_weighted'] = data['transition_signal'] * data['transition_confidence']
    data['combined_signal'] = data['high_weighted'] + data['low_weighted'] + data['transition_weighted']
    
    # Signal Quality Assessment
    data['momentum_range_consistency'] = data['momentum_regime'].rolling(window=10, min_periods=1).corr(data['range_momentum_regime'])
    data['range_impact_stability'] = data['range_multiimpact_width'].rolling(window=10, min_periods=1).std() / (data['range_multiimpact_width'].rolling(window=10, min_periods=1).mean() + 1e-8)
    data['signal_quality'] = data['momentum_range_consistency'] / (data['range_impact_stability'] + 0.001)
    
    # Final Alpha Generation
    data['raw_alpha'] = data['combined_signal'] * data['signal_quality']
    data['range_momentum_normalization'] = data['raw_alpha'] / (data['medium_price_momentum'] + 0.001)
    data['regime_alpha'] = data['range_momentum_normalization'] * data['range_impact_efficiency_score']
    
    data['range_impact_strength'] = data['impact_scaling_ratio'] * data['range_multiimpact_width']
    data['volume_range_confirmation'] = data['volume_range_impact'] * data['range_impact_corr']
    data['enhanced_alpha'] = data['regime_alpha'] * data['range_impact_strength'] * data['volume_range_confirmation']
    
    # Final regime-based adjustment
    high_impact_regime = data['enhanced_alpha'] * data['impact_scaling_ratio']
    low_impact_regime = data['enhanced_alpha'] / (data['impact_scaling_ratio'] + 0.001)
    
    data['final_alpha'] = np.where(data['impact_scaling_ratio'] > 1, high_impact_regime, low_impact_regime)
    
    return data['final_alpha']
