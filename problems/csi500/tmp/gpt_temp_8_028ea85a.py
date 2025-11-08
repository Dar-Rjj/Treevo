import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Liquidity Elasticity Momentum with Microstructure Absorption alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all required columns
    data['close_prev'] = data['close'].shift(1)
    data['amount_prev'] = data['amount'].shift(1)
    
    # 1. Multi-Scale Liquidity Elasticity Analysis
    # Price Elasticity Measurement
    data['elasticity'] = (data['high'] - data['low']) / (data['volume'] * np.abs(data['close'] - data['close_prev']))
    data['elasticity'] = data['elasticity'].replace([np.inf, -np.inf], np.nan)
    
    # Compute 5-day elasticity volatility
    data['elasticity_vol'] = data['elasticity'].rolling(window=5, min_periods=3).std()
    
    # Elasticity regime
    data['elasticity_ma_5'] = data['elasticity'].rolling(window=5, min_periods=3).mean()
    data['elasticity_regime'] = np.where(
        data['elasticity'] > data['elasticity_ma_5'], 1,  # High elasticity
        np.where(data['elasticity'] < data['elasticity_ma_5'], -1, 0)  # Low elasticity, Normal
    )
    
    # Elasticity Momentum Patterns
    data['elasticity_momentum_3d'] = (data['elasticity'] / data['elasticity'].shift(3)) - 1
    data['elasticity_momentum_8d'] = (data['elasticity'] / data['elasticity'].shift(8)) - 1
    data['elasticity_compression'] = np.sign(data['elasticity_momentum_3d']) * np.sign(data['elasticity_momentum_8d'])
    
    # 2. Bidirectional Liquidity Absorption Dynamics
    # Volume-Weighted Absorption Calculation
    def calculate_absorption(data, window=20):
        buy_absorption = []
        sell_absorption = []
        
        for i in range(len(data)):
            if i < window:
                buy_absorption.append(np.nan)
                sell_absorption.append(np.nan)
                continue
                
            window_data = data.iloc[i-window+1:i+1]
            
            # Buy-side absorption
            buy_mask = window_data['close'] > window_data['close_prev']
            if buy_mask.sum() > 0:
                buy_abs = (window_data.loc[buy_mask, 'volume'] * 
                          (window_data.loc[buy_mask, 'close'] - window_data.loc[buy_mask, 'close_prev'])).sum()
                buy_vol = window_data.loc[buy_mask, 'volume'].sum()
                buy_absorption.append(buy_abs / buy_vol if buy_vol > 0 else 0)
            else:
                buy_absorption.append(0)
            
            # Sell-side absorption
            sell_mask = window_data['close'] < window_data['close_prev']
            if sell_mask.sum() > 0:
                sell_abs = (window_data.loc[sell_mask, 'volume'] * 
                           np.abs(window_data.loc[sell_mask, 'close'] - window_data.loc[sell_mask, 'close_prev'])).sum()
                sell_vol = window_data.loc[sell_mask, 'volume'].sum()
                sell_absorption.append(sell_abs / sell_vol if sell_vol > 0 else 0)
            else:
                sell_absorption.append(0)
        
        return buy_absorption, sell_absorption
    
    buy_abs, sell_abs = calculate_absorption(data)
    data['buy_absorption'] = buy_abs
    data['sell_absorption'] = sell_abs
    
    # Absorption asymmetry
    data['absorption_asymmetry'] = (data['buy_absorption'] - data['sell_absorption']) / (
        data['buy_absorption'] + data['sell_absorption'] + 1e-8)
    
    # Absorption Momentum Detection
    data['absorption_diff_3d'] = data['absorption_asymmetry'] - data['absorption_asymmetry'].shift(3)
    data['absorption_diff_8d'] = data['absorption_asymmetry'] - data['absorption_asymmetry'].shift(8)
    
    # Absorption regime
    data['absorption_ma_20'] = data['absorption_asymmetry'].rolling(window=20, min_periods=10).mean()
    data['absorption_regime'] = (data['absorption_asymmetry'] > data['absorption_ma_20']).astype(int)
    
    # 3. Order Flow Velocity Integration
    # Amount Pressure Dynamics
    data['amount_velocity'] = (data['amount'] - data['amount_prev']) / (data['amount_prev'] + 1e-8)
    data['amount_acceleration'] = data['amount_velocity'] - data['amount_velocity'].shift(5)
    data['velocity_clustering'] = np.sign(data['amount_velocity']) * np.sign(data['amount_velocity'].shift(1))
    
    # Velocity-Microstructure Alignment
    def calculate_velocity_persistence(data, window=5):
        persistence = []
        for i in range(len(data)):
            if i < window:
                persistence.append(np.nan)
                continue
                
            window_vel = data['amount_velocity'].iloc[i-window+1:i+1]
            signs = np.sign(window_vel)
            if len(signs) == 0:
                persistence.append(0)
                continue
                
            current_sign = signs.iloc[-1]
            count = 0
            for j in range(len(signs)-1, -1, -1):
                if signs.iloc[j] == current_sign:
                    count += 1
                else:
                    break
            persistence.append(count)
        return persistence
    
    data['velocity_persistence'] = calculate_velocity_persistence(data)
    data['velocity_exhaustion'] = data['velocity_persistence'] * np.abs(data['amount_velocity'])
    data['velocity_micro_convergence'] = np.sign(data['amount_velocity']) * np.sign(data['absorption_asymmetry'])
    
    # 4. Microstructure State Classification
    # Three-State Market Microstructure
    data['microstructure_state'] = np.where(
        (data['elasticity_regime'] == 1) & (data['absorption_asymmetry'] < data['absorption_ma_20']),  # Fragile
        -1,
        np.where(
            (data['elasticity_regime'] == -1) & (data['absorption_asymmetry'] > data['absorption_ma_20']),  # Resilient
            1,
            0  # Transitional
        )
    )
    
    # State Duration Analysis
    def calculate_state_duration(data):
        duration = []
        current_state = None
        current_duration = 0
        
        for state in data['microstructure_state']:
            if pd.isna(state):
                duration.append(np.nan)
                continue
                
            if state == current_state:
                current_duration += 1
            else:
                current_state = state
                current_duration = 1
            duration.append(current_duration)
        return duration
    
    data['state_duration'] = calculate_state_duration(data)
    data['state_transition_prob'] = 1 / (data['state_duration'] + 1)
    data['state_exhaustion'] = (data['state_duration'] > 5).astype(int)
    
    # 5. Elasticity-Absorption Signal Fusion
    # Core Elasticity Component
    data['base_elasticity_signal'] = (data['elasticity_momentum_3d'] - data['elasticity_momentum_8d']) * data['elasticity']
    data['elasticity_regime_adj'] = data['base_elasticity_signal'] * (1 + data['elasticity_compression'])
    data['volatility_scaled_elasticity'] = data['elasticity_regime_adj'] / (1 + data['elasticity_vol'])
    
    # Absorption Enhancement
    data['absorption_momentum_signal'] = (data['absorption_diff_3d'] - data['absorption_diff_8d']) * data['absorption_asymmetry']
    data['regime_transition_boost'] = data['absorption_momentum_signal'] * data['state_transition_prob']
    data['combined_elasticity_absorption'] = data['volatility_scaled_elasticity'] * data['absorption_momentum_signal']
    
    # 6. Velocity-Adaptive Signal Refinement
    # Velocity Confirmation Mechanism
    data['velocity_weighted_signal'] = data['combined_elasticity_absorption'] * data['amount_velocity']
    data['persistence_amplified'] = data['velocity_weighted_signal'] * (1 + data['velocity_persistence'] / 5)
    data['velocity_convergence_adj'] = data['persistence_amplified'] * (1 + data['velocity_micro_convergence'])
    
    # State-Dependent Signal Optimization
    data['fragile_signal'] = data['velocity_convergence_adj'] * data['elasticity']
    data['fragile_signal'] = data['fragile_signal'] * (1 - data['state_exhaustion'])
    
    data['resilient_signal'] = data['velocity_convergence_adj'] * data['absorption_asymmetry']
    data['resilient_signal'] = data['resilient_signal'] * (1 + data['absorption_momentum_signal'])
    
    data['transitional_signal'] = data['velocity_convergence_adj'] * (data['elasticity'] + data['absorption_asymmetry']) / 2
    data['transitional_signal'] = data['transitional_signal'] * (1 + np.abs(data['amount_velocity']))
    
    # 7. Final Alpha Construction
    # Multi-Scale Momentum Integration
    data['short_term_component'] = data['elasticity_momentum_3d'] * data['absorption_diff_3d']
    data['medium_term_component'] = data['elasticity_momentum_8d'] * data['absorption_diff_8d']
    data['momentum_convergence'] = data['short_term_component'] - data['medium_term_component']
    
    # Microstructure-Adaptive Output
    data['fragile_alpha'] = data['fragile_signal'] * data['momentum_convergence']
    data['resilient_alpha'] = data['resilient_signal'] * data['momentum_convergence']
    data['transitional_alpha'] = data['transitional_signal'] * data['momentum_convergence']
    
    # Unified alpha factor
    def calculate_state_probability(data, window=10):
        state_probs = []
        for i in range(len(data)):
            if i < window:
                state_probs.append({'fragile': 0.33, 'resilient': 0.33, 'transitional': 0.34})
                continue
                
            window_states = data['microstructure_state'].iloc[i-window+1:i+1]
            fragile_count = (window_states == -1).sum()
            resilient_count = (window_states == 1).sum()
            transitional_count = (window_states == 0).sum()
            total = fragile_count + resilient_count + transitional_count
            
            if total > 0:
                state_probs.append({
                    'fragile': fragile_count / total,
                    'resilient': resilient_count / total,
                    'transitional': transitional_count / total
                })
            else:
                state_probs.append({'fragile': 0.33, 'resilient': 0.33, 'transitional': 0.34})
        return state_probs
    
    state_probs = calculate_state_probability(data)
    
    # Calculate final unified alpha
    unified_alpha = []
    for i, probs in enumerate(state_probs):
        if pd.isna(data['fragile_alpha'].iloc[i]) or pd.isna(data['resilient_alpha'].iloc[i]) or pd.isna(data['transitional_alpha'].iloc[i]):
            unified_alpha.append(np.nan)
        else:
            weighted_alpha = (probs['fragile'] * data['fragile_alpha'].iloc[i] +
                            probs['resilient'] * data['resilient_alpha'].iloc[i] +
                            probs['transitional'] * data['transitional_alpha'].iloc[i])
            unified_alpha.append(weighted_alpha)
    
    # Return the final alpha factor series
    alpha_series = pd.Series(unified_alpha, index=data.index)
    return alpha_series
