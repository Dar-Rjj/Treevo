import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Intraday Volatility Signature
    data['true_range_norm'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['gap_volatility'] = np.abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['close_to_close_vol'] = np.abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volume-Volatility Coupling
    data['volume_vol_ratio'] = data['volume'] / (data['high'] - data['low'])
    data['amount_density'] = data['amount'] / data['volume']
    data['volume_spike'] = data['volume'] / data['volume'].shift(1)
    
    # Regime Classification
    data['true_range_avg_4d'] = data['true_range_norm'].rolling(window=4, min_periods=1).mean().shift(1)
    data['high_vol_regime'] = data['true_range_norm'] > data['true_range_avg_4d']
    data['low_vol_regime'] = data['true_range_norm'] < data['true_range_avg_4d']
    data['transition_regime'] = data['gap_volatility'] > data['close_to_close_vol']
    
    # Adaptive Momentum Framework
    # Price Momentum Components
    data['raw_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Volume-Weighted Momentum
    data['volume_adj_return'] = (data['close'] - data['open']) / data['volume']
    data['amount_efficiency'] = (data['close'] - data['open']) / data['amount']
    data['volume_flow'] = data['volume_spike'] * (data['close'] - data['open'])
    
    # Regime-Specific Momentum
    data['high_vol_momentum'] = data['raw_momentum'] * data['volume_vol_ratio']
    data['low_vol_momentum'] = data['intraday_momentum'] * data['amount_density']
    data['transition_momentum'] = data['gap_momentum'] * data['volume_spike']
    
    # Volatility-Adjusted Signal Generation
    # Signal Quality Assessment
    data['momentum_consistency'] = np.sign(data['raw_momentum']) * np.sign(data['intraday_momentum'])
    data['volume_confirmation'] = data['volume_spike'] * np.sign(data['close'] - data['open'])
    data['gap_alignment'] = np.sign(data['gap_momentum']) * np.sign(data['intraday_momentum'])
    
    # Volatility Dampening
    data['high_vol_dampener'] = data['raw_momentum'] / data['true_range_norm']
    data['low_vol_amplifier'] = data['intraday_momentum'] * data['amount_density']
    data['transition_stabilizer'] = data['gap_momentum'] * data['volume_spike']
    
    # Signal Integration
    data['core_signal'] = data['momentum_consistency'] * data['volume_confirmation']
    data['vol_adapted_signal'] = data['core_signal'] * np.where(
        data['high_vol_regime'], data['high_vol_dampener'],
        np.where(data['low_vol_regime'], data['low_vol_amplifier'], data['transition_stabilizer'])
    )
    data['enhanced_signal'] = data['vol_adapted_signal'] * data['gap_alignment']
    
    # Multi-Timeframe Volatility Structure
    # Short-term Volatility Patterns
    data['vol_acceleration'] = data['true_range_norm'] / data['true_range_norm'].shift(1)
    data['vol_vol_correlation'] = data['volume_vol_ratio'] / data['volume_vol_ratio'].shift(1)
    data['gap_vol_momentum'] = data['gap_volatility'] / data['gap_volatility'].shift(1)
    
    # Medium-term Volatility Memory
    data['vol_persistence'] = data['true_range_norm'].rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x.iloc[1:] > x.iloc[0]) if len(x) > 1 else 0, raw=False
    )
    data['volume_regime_stability'] = data['volume'] / data['volume'].rolling(window=4, min_periods=1).mean().shift(1)
    data['amount_consistency'] = data['amount_density'] / data['amount_density'].shift(1)
    
    # Volatility Structure Integration
    data['short_term_component'] = data['vol_acceleration'] * data['vol_vol_correlation']
    data['medium_term_component'] = data['vol_persistence'] * data['volume_regime_stability']
    data['structural_vol_score'] = data['short_term_component'] * data['medium_term_component']
    
    # Regime Transition Detection
    # Transition Signals
    data['vol_breakout'] = data['true_range_norm'] > (2 * data['true_range_norm'].shift(1))
    data['volume_regime_shift'] = data['volume'] > (2 * data['volume'].shift(1))
    data['amount_anomaly'] = data['amount_density'] > (2 * data['amount_density'].shift(1))
    
    # Transition Confirmation
    data['multi_signal_agreement'] = data['vol_breakout'].astype(float) * data['volume_regime_shift'].astype(float)
    data['momentum_transition'] = data['raw_momentum'] * data['intraday_momentum']
    data['volume_transition'] = data['volume_spike'] * data['amount_density']
    
    # Transition Integration
    data['transition_score'] = data['multi_signal_agreement'] * data['momentum_transition']
    data['confirmed_transition'] = data['transition_score'] * data['volume_transition']
    data['transition_impact'] = data['confirmed_transition'] * data['structural_vol_score']
    
    # Adaptive Alpha Synthesis
    # Regime-Specific Alpha Components
    data['high_vol_alpha'] = data['high_vol_momentum'] * data['high_vol_dampener']
    data['low_vol_alpha'] = data['low_vol_momentum'] * data['low_vol_amplifier']
    data['transition_alpha'] = data['transition_momentum'] * data['transition_stabilizer']
    
    # Volatility Structure Integration
    data['core_alpha'] = data['enhanced_signal'] * data['structural_vol_score']
    data['transition_enhanced_alpha'] = data['core_alpha'] * data['transition_impact']
    
    # Regime-Adapted Alpha
    data['regime_adapted_alpha'] = np.where(
        data['high_vol_regime'], data['high_vol_alpha'],
        np.where(data['low_vol_regime'], data['low_vol_alpha'], data['transition_alpha'])
    )
    
    # Final Alpha Construction
    data['primary_alpha'] = data['regime_adapted_alpha'] * data['transition_enhanced_alpha']
    data['quality_filter'] = data['primary_alpha'] * data['momentum_consistency']
    data['volume_confirmed_alpha'] = data['quality_filter'] * data['volume_confirmation']
    
    # Adaptive Alpha Output
    data['final_signal'] = data['volume_confirmed_alpha'] * data['structural_vol_score']
    data['regime_confidence'] = data['final_signal'] * data['vol_persistence']
    data['transition_stability'] = data['regime_confidence'] * data['transition_score']
    
    # Return the final alpha factor
    return data['transition_stability']
