import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic price and volume changes
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['volume_ret'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Asymmetric Quantum State Analysis
    # Volatility-Entanglement Asymmetry
    up_mask = data['close'] > data['open']
    down_mask = data['close'] < data['open']
    
    # Calculate rolling correlations for up and down days
    up_entanglement = pd.Series(index=data.index, dtype=float)
    down_entanglement = pd.Series(index=data.index, dtype=float)
    
    for i in range(2, len(data)):
        if i >= 3:
            window_data = data.iloc[i-2:i+1]
            up_window = window_data[window_data['close'] > window_data['open']]
            down_window = window_data[window_data['close'] < window_data['open']]
            
            if len(up_window) >= 2:
                up_entanglement.iloc[i] = up_window['close_ret'].corr(up_window['volume_ret'])
            if len(down_window) >= 2:
                down_entanglement.iloc[i] = down_window['close_ret'].corr(down_window['volume_ret'])
    
    entanglement_asymmetry = up_entanglement / down_entanglement.replace(0, np.nan)
    
    # Path-Efficiency Quantum States
    data['price_range'] = data['high'] - data['low']
    data['intraday_move'] = data['close'] - data['open']
    
    # Calculate volume-weighted price movement (approximation)
    data['vw_price_movement'] = data['volume'] * data['price_range']
    data['rolling_vw_movement'] = data['vw_price_movement'].rolling(window=5, min_periods=1).sum()
    
    quantum_path_efficiency = data['intraday_move'] / data['rolling_vw_movement'].replace(0, np.nan)
    
    # Calculate phase coherence (price-volume correlation over 5 days)
    phase_coherence = data['close_ret'].rolling(window=5).corr(data['volume_ret'])
    
    up_quantum_coherence = quantum_path_efficiency * phase_coherence
    down_quantum_decoherence = quantum_path_efficiency * (1 - abs(phase_coherence))
    
    # Asymmetric State Transitions
    # Mixed state approximation using price-volume relationship
    mixed_state = data['close_ret'] * data['volume_ret']
    mixed_state_change = mixed_state.diff()
    
    up_state_momentum = mixed_state_change * data['volume']
    down_state_momentum = mixed_state_change * data['volume']
    
    state_transition_asymmetry = up_state_momentum / down_state_momentum.replace(0, np.nan)
    
    # Asymmetric Quantum Interference Framework
    # Regime-Dependent Quantum Interference
    volatility_regime = (data['high'] - data['low']) / data['close'].shift(1)
    volume_regime = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    
    interference_regime_asymmetry = volatility_regime / volume_regime.replace(0, np.nan)
    
    # Asymmetric Tunneling Effects
    up_tunneling = pd.Series(index=data.index, dtype=float)
    down_tunneling = pd.Series(index=data.index, dtype=float)
    
    for i in range(3, len(data)):
        if up_mask.iloc[i]:
            prev_highs = data['high'].iloc[i-3:i]
            if len(prev_highs) > 0:
                up_tunneling.iloc[i] = (data['high'].iloc[i] - prev_highs.max()) / (data['high'].iloc[i] - data['low'].iloc[i])
        
        if down_mask.iloc[i]:
            prev_lows = data['low'].iloc[i-3:i]
            if len(prev_lows) > 0:
                down_tunneling.iloc[i] = (prev_lows.min() - data['low'].iloc[i]) / (data['high'].iloc[i] - data['low'].iloc[i])
    
    tunneling_asymmetry = up_tunneling / down_tunneling.replace(0, np.nan)
    
    # Entangled Asymmetric Momentum Generation
    # Quantum Asymmetric Velocity
    up_quantum_velocity = data['close_ret'] * data['volume'] * up_entanglement
    down_quantum_velocity = data['close_ret'] * data['volume'] * down_entanglement
    velocity_asymmetry_momentum = up_quantum_velocity / down_quantum_velocity.replace(0, np.nan)
    
    # Regime-Adaptive Quantum Momentum
    volatility_quantum_momentum = velocity_asymmetry_momentum * volatility_regime
    volume_quantum_momentum = data['close_ret'].diff() * volume_regime  # Acceleration approximation
    regime_quantum_alignment = volatility_quantum_momentum * volume_quantum_momentum
    
    # Quantum Asymmetric Filtering
    # Decoherence-Based Asymmetry Filtering
    price_volume_decoherence = 1 - abs(phase_coherence)
    
    up_decoherence_adj = quantum_path_efficiency / (1 + abs(price_volume_decoherence))
    down_decoherence_adj = quantum_path_efficiency / (1 + abs(price_volume_decoherence))
    asymmetric_decoherence_filter = up_decoherence_adj / down_decoherence_adj.replace(0, np.nan)
    
    # State Transition Asymmetry Filtering
    pure_state = abs(data['close_ret'])  # Pure state approximation
    pure_state_asymmetric = pure_state * velocity_asymmetry_momentum
    mixed_state_asymmetric = mixed_state * state_transition_asymmetry
    transition_regime_filter = pure_state_asymmetric * mixed_state_asymmetric
    
    # Interference Asymmetry Filtering
    volatility_interference_filter = interference_regime_asymmetry * volatility_quantum_momentum
    volume_interference_filter = (up_quantum_coherence + down_quantum_decoherence) * volume_quantum_momentum
    multi_regime_interference = volatility_interference_filter * volume_interference_filter
    
    # Quantum Asymmetric Alpha Synthesis
    core_quantum_asymmetry = velocity_asymmetry_momentum * entanglement_asymmetry * state_transition_asymmetry
    regime_quantum_efficiency = regime_quantum_alignment * quantum_path_efficiency * asymmetric_decoherence_filter
    quantum_asymmetric_momentum = core_quantum_asymmetry * regime_quantum_efficiency * multi_regime_interference
    
    # Final alpha factor
    alpha_factor = quantum_asymmetric_momentum.fillna(0)
    
    return alpha_factor
