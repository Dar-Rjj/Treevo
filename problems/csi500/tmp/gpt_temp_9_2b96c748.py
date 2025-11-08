import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Amplitude-Volume Divergence Factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Multi-Scale Amplitude & Momentum Analysis
    # Amplitude Structure Assessment
    data['up_amplitude'] = (data['high'] - data['open']) / data['open']
    data['down_amplitude'] = (data['open'] - data['low']) / data['open']
    
    # Short-term upward amplitude (geometric mean from t-1 to t)
    data['short_up_amp'] = np.exp((np.log(data['up_amplitude'].shift(1) + np.log(data['up_amplitude'])) / 2))
    
    # Short-term downward amplitude (geometric mean from t-1 to t)
    data['short_down_amp'] = np.exp((np.log(data['down_amplitude'].shift(1) + np.log(data['down_amplitude'])) / 2))
    
    # Amplitude asymmetry
    data['amp_asymmetry'] = data['up_amplitude'] - data['down_amplitude']
    
    # Medium-term amplitude asymmetry (geometric mean from t-5 to t)
    log_amp_asym = np.log(data['amp_asymmetry'].rolling(window=6).apply(lambda x: np.prod(x + 1) - 1, raw=False))
    data['medium_amp_asym'] = np.exp(log_amp_asym / 6)
    
    # Amplitude ratio for regime classification
    data['amp_ratio'] = (data['short_up_amp'] + data['short_down_amp']) / (np.abs(data['medium_amp_asym']) + 1e-8)
    
    # Momentum State Classification
    data['intraday_reversal'] = np.sign(data['close'] - data['open']) * (data['high'] - data['low']) / data['open']
    
    # Multi-day reversal
    data['multi_day_reversal'] = (data['close'] - data['close'].shift(3)) / (data['high'].shift(3) - data['low'].shift(3) + 1e-8)
    
    # Reversal acceleration
    data['reversal_accel'] = data['multi_day_reversal'] - data['multi_day_reversal'].shift(2)
    
    # Regime Detection System
    data['regime'] = 'transition'
    data.loc[data['amp_ratio'] > 1.3, 'regime'] = 'high'
    data.loc[data['amp_ratio'] < 0.8, 'regime'] = 'low'
    
    # 2. Volume-Amplitude Divergence Framework
    # Directional Volume Analysis
    data['up_volume'] = data['volume'] * (data['amp_asymmetry'] > 0)
    data['down_volume'] = data['volume'] * (data['amp_asymmetry'] < 0)
    
    # Cumulative volume on days with positive/negative amplitude asymmetry
    data['cum_up_volume'] = data['up_volume'].rolling(window=5, min_periods=1).sum()
    data['cum_down_volume'] = data['down_volume'].rolling(window=5, min_periods=1).sum()
    
    # Volume imbalance
    data['volume_imbalance'] = data['cum_up_volume'] / (data['cum_down_volume'] + 1e-8)
    
    # Amplitude-Momentum Strength Assessment
    # Amplitude efficiency
    data['amp_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)).rolling(window=3).mean()
    
    # Reversal persistence (count of consecutive same-direction intraday reversals)
    data['reversal_dir'] = np.sign(data['intraday_reversal'])
    data['reversal_persistence'] = data['reversal_dir'].rolling(window=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x[i] == x[i-1] and x[i] != 0]), raw=True
    )
    
    # Volume-weighted amplitude
    data['volume_weighted_amp'] = data['amp_asymmetry'] * (data['volume'] / data['volume'].rolling(window=10).mean())
    
    # Divergence Detection
    # Amplitude-volume correlation (5-day correlation)
    data['amp_volume_corr'] = data['amp_asymmetry'].rolling(window=5).corr(data['volume'])
    
    # Reversal-volume alignment
    data['reversal_volume_alignment'] = np.sign(data['intraday_reversal']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Divergence strength
    data['divergence_strength'] = np.abs(data['amp_asymmetry'] - data['volume_weighted_amp'])
    
    # 3. Regime-Adaptive Signal Processing
    # Initialize regime-specific signals
    data['high_amp_signal'] = 0.0
    data['low_amp_signal'] = 0.0
    data['transition_signal'] = 0.0
    
    # High Amplitude Signal Enhancement
    high_mask = data['regime'] == 'high'
    data.loc[high_mask, 'high_amp_signal'] = (
        data.loc[high_mask, 'volume_imbalance'] * 
        data.loc[high_mask, 'divergence_strength'] * 
        data.loc[high_mask, 'reversal_accel']
    )
    
    # Low Amplitude Signal Refinement
    low_mask = data['regime'] == 'low'
    data.loc[low_mask, 'low_amp_signal'] = (
        data.loc[low_mask, 'reversal_persistence'] * 
        np.sign(data.loc[low_mask, 'volume'] - data.loc[low_mask, 'volume'].shift(1)) *
        data.loc[low_mask, 'amp_efficiency']
    )
    
    # Transition Regime Signal Balancing
    trans_mask = data['regime'] == 'transition'
    data.loc[trans_mask, 'transition_signal'] = (
        0.5 * data.loc[trans_mask, 'divergence_strength'] +
        0.3 * data.loc[trans_mask, 'reversal_accel'] +
        0.2 * data.loc[trans_mask, 'reversal_volume_alignment']
    )
    
    # 4. Adaptive Factor Construction
    # Multi-Timeframe Signal Integration
    data['short_term_signal'] = (
        data['intraday_reversal'] * 
        data['amp_efficiency'] * 
        data['reversal_volume_alignment']
    ).rolling(window=2).mean()
    
    data['medium_term_signal'] = (
        data['multi_day_reversal'] * 
        data['volume_imbalance'] * 
        data['amp_volume_corr']
    ).rolling(window=6).mean()
    
    # Regime-weighted combination
    regime_weights = {
        'high': [0.6, 0.4],  # [short_term, medium_term]
        'low': [0.3, 0.7],
        'transition': [0.5, 0.5]
    }
    
    data['regime_weighted_signal'] = 0.0
    for regime, weights in regime_weights.items():
        mask = data['regime'] == regime
        data.loc[mask, 'regime_weighted_signal'] = (
            weights[0] * data.loc[mask, 'short_term_signal'] +
            weights[1] * data.loc[mask, 'medium_term_signal']
        )
    
    # Volume-Confirmed Amplitude Signals
    data['volume_aligned_amp'] = data['amp_asymmetry'] * data['reversal_volume_alignment']
    data['divergence_corrected_reversal'] = data['intraday_reversal'] / (data['divergence_strength'] + 1e-8)
    
    # 5. Composite Factor Generation
    # Amplitude-Normalized Divergence Score
    regime_scaling = {
        'high': 1.5,
        'low': 0.7,
        'transition': 1.0
    }
    
    data['regime_scale'] = data['regime'].map(regime_scaling)
    data['amp_normalized_divergence'] = data['divergence_strength'] * data['regime_scale']
    
    # Reversal-Volume Alignment Factor
    data['reversal_volume_factor'] = (
        data['reversal_accel'] * 
        np.sign(data['volume'] - data['volume'].shift(1)) *
        data['amp_efficiency']
    )
    
    # Final Regime-Adaptive Predictive Signal
    data['composite_factor'] = (
        0.4 * data['regime_weighted_signal'] +
        0.3 * data['amp_normalized_divergence'] +
        0.3 * data['reversal_volume_factor']
    )
    
    # Apply regime-specific final adjustments
    data.loc[data['regime'] == 'high', 'composite_factor'] *= 1.2
    data.loc[data['regime'] == 'low', 'composite_factor'] *= 0.8
    
    # Return the final factor series
    return data['composite_factor']
