import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Quantum Temporal Asymmetry Alpha Factor
    """
    data = df.copy()
    
    # Basic calculations
    data['close_ret_1'] = data['close'].pct_change(1)
    data['close_ret_2'] = data['close'].pct_change(2)
    data['close_ret_5'] = data['close'].pct_change(5)
    data['close_ret_10'] = data['close'].pct_change(10)
    data['volume_ret_1'] = data['volume'].pct_change(1)
    
    # Rolling calculations
    data['volume_ma_5'] = data['volume'].rolling(5).mean()
    data['volume_ma_10'] = data['volume'].rolling(10).mean()
    data['close_ma_5'] = data['close'].rolling(5).mean()
    data['high_low_ma_5'] = (data['high'] - data['low']).rolling(5).mean()
    data['high_low_ma_10'] = (data['high'] - data['low']).rolling(10).mean()
    data['close_std_5'] = data['close'].rolling(5).std()
    data['close_std_10'] = data['close'].rolling(10).std()
    data['open_ma_3'] = data['open'].rolling(3).mean()
    data['open_std_3'] = data['open'].rolling(3).std()
    
    # Quantum Momentum State Detection
    # Temporal Momentum Entanglement
    data['quantum_momentum_div'] = (
        (data['close_ret_1'] / data['close_ret_2'].shift(1) - 
         data['close_ret_5'] / data['close_ret_10'].shift(5)) * 
        (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    data['phase_coherent_momentum'] = (
        np.sign(data['close_ret_1']) * np.sign(data['volume_ret_1']) * 
        abs(data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    data['quantum_collapse_momentum'] = (
        abs(data['close'] - data['close'].shift(1)) * data['volume'] * 
        (1 - (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, 1))
    ).fillna(0)
    
    # Temporal Persistence Patterns
    close_sign = np.sign(data['close'] - data['close'].shift(1))
    data['momentum_duration_strength'] = (
        close_sign.rolling(3).apply(lambda x: (x == x.shift(1)).sum(), raw=False) * 
        abs(data['close'] - data['close'].shift(1))
    ).fillna(0)
    
    volume_sign = np.sign(data['volume'] - data['volume'].shift(1))
    data['quantum_momentum_consistency'] = (
        close_sign.rolling(5).corr(volume_sign) * data['volume']
    ).fillna(0)
    
    data['entangled_persistence'] = (
        ((data['close'] - data['close'].shift(1)) * volume_sign)
        .rolling(3).sum() * (data['close'] - data['close'].shift(1))
    ).fillna(0)
    
    # Asymmetric Quantum Patterns
    data['gap_driven_quantum_momentum'] = (
        ((data['open'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, 1)) * 
        ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)) * 
        data['volume']
    ).fillna(0)
    
    data['intraday_quantum_capture'] = (
        ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)) * 
        (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    data['overnight_quantum_persistence'] = (
        (data['open'] - data['close'].shift(1)) * 
        (data['close'] - data['open']) * data['volume']
    ).fillna(0)
    
    # Quantum Range Efficiency Framework
    data['daily_quantum_efficiency'] = (
        ((data['close'] - data['open']) / (data['high'] - data['low']).replace(0, 1)) * 
        data['volume'] * 
        (1 - (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, 1))
    ).fillna(0)
    
    data['multi_day_quantum_compression'] = (
        ((data['high'] - data['low']) / data['high_low_ma_5']) * 
        np.sign(data['close'] - data['close'].shift(1)) * 
        (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    # Quantum Breakout Potential
    high_roll = data['high'].rolling(3)
    low_roll = data['low'].rolling(3)
    data['quantum_breakout_potential'] = (
        ((data['high'] - high_roll.apply(lambda x: x.iloc[:-1].max() if len(x) > 1 else np.nan, raw=False)) / 
         (data['high'] - data['low']).replace(0, 1) + 
         (low_roll.apply(lambda x: x.iloc[:-1].min() if len(x) > 1 else np.nan, raw=False) - data['low']) / 
         (data['high'] - data['low']).replace(0, 1)) * data['volume']
    ).fillna(0)
    
    # Quantum Price-Level Efficiency
    data['quantum_support_resistance'] = (
        (data['close'] > data['close_ma_5']).rolling(3).mean() * 
        (data['close'] - data['low'].rolling(4).min()) * data['volume']
    ).fillna(0)
    
    data['quantum_price_rejection'] = (
        (abs(data['high'] - data[['open', 'close']].max(axis=1)) / 
         (data['high'] - data['low']).replace(0, 1) + 
         abs(data['low'] - data[['open', 'close']].min(axis=1)) / 
         (data['high'] - data['low']).replace(0, 1)) * data['volume']
    ).fillna(0)
    
    data['quantum_opening_anchoring'] = (
        ((data['open'] - data['open_ma_3']) / data['open_std_3'].replace(0, 1)) * 
        data['volume'] * np.sign(data['close'] - data['open'])
    ).fillna(0)
    
    # Temporal Quantum Patterns
    data['quantum_range_expansion'] = (
        ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)).replace(0, 1)) * 
        (data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    data['quantum_range_contraction'] = (
        ((data['high'].shift(1) - data['low'].shift(1)) / (data['high'] - data['low']).replace(0, 1)) * 
        np.sign(data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    data['quantum_volatility_regime'] = (
        (data['close_std_5'] / (data['high'] - data['low']).replace(0, 1)) * 
        (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    # Quantum Volume-Timing Analysis
    data['quantum_volume_acceleration'] = (
        (data['volume'] / data['volume'].shift(1)) * 
        (data['volume'].shift(1) / data['volume'].shift(2)) * 
        (data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    data['quantum_volume_clustering'] = (
        (data['volume'] > data['volume_ma_5']).rolling(3).mean() * 
        (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    # Quantum Volume-Weighted Timing
    data['quantum_volume_weighted_timing'] = (
        ((data['volume'] * (data['close'] - data['open'])).rolling(3).sum() / 
         data['volume'].rolling(3).sum().replace(0, 1)) * 
        data['volume'] * np.sign(data['close'] - data['open'])
    ).fillna(0)
    
    # Quantum Volume Transition
    data['quantum_volume_transition'] = (
        (data['volume'] / data['volume_ma_10']) * 
        (data['volume'].shift(1) / data['volume_ma_10'].shift(1)) * 
        (data['close'] - data['close'].shift(1))
    ).fillna(0)
    
    # Quantum Regime Detection
    trend_strength = close_sign.rolling(5).mean().abs()
    data['quantum_trend_classification'] = (
        trend_strength * (data['volume'] / data['volume_ma_5']) * 
        (data['close'] - data['close'].shift(1))
    ).fillna(0)
    
    # Quantum Volatility-Momentum
    data['quantum_high_volatility'] = (
        (data['close_std_5'] / data['close_std_5'].rolling(10).mean().replace(0, 1)) * 
        (data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    data['quantum_low_volatility_breakout'] = (
        ((data['high'] - data['low']) / data['high_low_ma_10']) * 
        (data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    data['quantum_volatility_compression'] = (
        (data['close_std_5'] / data['close_std_10'].replace(0, 1)) * 
        (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    # Quantum Volume-Momentum Alignment
    data['quantum_volume_momentum_divergence'] = (
        (np.sign(data['volume'] - data['volume'].shift(1)) != np.sign(data['close'] - data['close'].shift(1))) * 
        abs(data['close'] - data['close'].shift(1)) * data['volume']
    ).fillna(0)
    
    # Quantum Asymmetric Signal Generation
    # Quantum Temporal Signals
    data['quantum_short_term_acceleration'] = (
        (data['close_ret_1'] / data['close_ret_2'].shift(1)) * 
        data['volume'] * data['phase_coherent_momentum']
    ).fillna(0)
    
    data['quantum_medium_term_persistence'] = (
        (data['close_ret_5'] / data['close_ret_10'].shift(5)) * 
        (data['volume'] / data['volume_ma_5']) * data['quantum_momentum_consistency']
    ).fillna(0)
    
    data['quantum_regime_strength'] = (
        data['quantum_trend_classification'] * data['volume'] * 
        (data['close'] - data['close'].shift(1)) * data['entangled_persistence']
    ).fillna(0)
    
    # Quantum Range Signals
    data['quantum_high_efficiency'] = (
        (data['daily_quantum_efficiency'] > 0.7) * data['volume'] * data['quantum_breakout_potential']
    ).fillna(0)
    
    data['quantum_low_efficiency'] = (
        (data['daily_quantum_efficiency'] < 0.3) * data['volume'] * data['quantum_range_contraction']
    ).fillna(0)
    
    data['quantum_range_expansion_signal'] = (
        data['quantum_range_expansion'] * (data['volume'] / data['volume_ma_5']) * 
        data['quantum_volatility_regime']
    ).fillna(0)
    
    # Quantum Volume Signals
    data['quantum_volume_acceleration_signal'] = (
        data['quantum_volume_acceleration'] * (data['close'] - data['close'].shift(1)) * 
        data['quantum_volume_transition']
    ).fillna(0)
    
    data['quantum_volume_regime'] = (
        data['quantum_volume_transition'] * (data['close'] - data['close'].shift(1)) * 
        data['quantum_volume_momentum_divergence']
    ).fillna(0)
    
    # Quantum Adaptive Weighting
    # Quantum Trend Weights
    data['quantum_strong_multiplier'] = (
        1.5 * data['quantum_trend_classification'] * (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    data['quantum_weak_multiplier'] = (
        0.7 * (1 - abs(data['quantum_trend_classification'])) * (data['volume'] / data['volume_ma_5'])
    ).fillna(0)
    
    data['quantum_transition_multiplier'] = (
        1.0 * (data['volume'] / data['volume_ma_5']) * data['quantum_volume_clustering']
    ).fillna(0)
    
    # Quantum Volatility Weights
    data['quantum_high_volatility_amplifier'] = (
        1.3 * (data['close_std_5'] / data['close_std_5'].rolling(10).mean().replace(0, 1)) * data['volume']
    ).fillna(0)
    
    data['quantum_low_volatility_attenuator'] = (
        0.8 * (data['close_std_5'].rolling(10).mean() / data['close_std_5'].replace(0, 1)) * data['volume']
    ).fillna(0)
    
    data['quantum_normal_volatility'] = (
        1.0 * (data['volume'] / data['volume_ma_5']) * data['quantum_volatility_compression']
    ).fillna(0)
    
    # Quantum Volume Weights
    data['quantum_high_volume_confidence'] = (
        1.4 * (data['volume'] / data['volume_ma_10']) * data['quantum_volume_clustering']
    ).fillna(0)
    
    data['quantum_low_volume_caution'] = (
        0.6 * (data['volume_ma_10'] / data['volume'].replace(0, 1)) * data['quantum_volume_momentum_divergence']
    ).fillna(0)
    
    data['quantum_normal_volume'] = (
        1.0 * data['quantum_volume_clustering'] * data['quantum_volume_transition']
    ).fillna(0)
    
    # Final Quantum Asymmetry Factor
    # Core Components
    data['quantum_momentum_core'] = (
        data['quantum_short_term_acceleration'] * 
        data['quantum_medium_term_persistence'] * 
        data['quantum_volume_acceleration_signal']
    ).fillna(0)
    
    data['quantum_range_core'] = (
        data['quantum_high_efficiency'] * 
        data['quantum_range_expansion_signal'] * 
        (1 - data['quantum_price_rejection'])
    ).fillna(0)
    
    data['quantum_volume_core'] = (
        data['quantum_volume_acceleration_signal'] * 
        data['quantum_volume_regime'] * 
        data['quantum_volume_clustering']
    ).fillna(0)
    
    # Final Factor Calculation
    data['quantum_weighted_asymmetry'] = (
        (data['quantum_momentum_core'] * data['quantum_strong_multiplier']) +
        (data['quantum_range_core'] * data['quantum_high_volatility_amplifier']) +
        (data['quantum_volume_core'] * data['quantum_high_volume_confidence'])
    ).fillna(0)
    
    data['quantum_temporal_convergence'] = (
        data['quantum_weighted_asymmetry'] * 
        data['quantum_regime_strength'] * 
        (1 + data['quantum_breakout_potential'])
    ).fillna(0)
    
    return data['quantum_temporal_convergence']
