import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Regime-Sensitive Momentum alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Framework
    # Multi-Timeframe Price Efficiency
    data['short_term_efficiency'] = abs(data['close'] - data['close'].shift(2)) / (
        data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    )
    
    data['medium_term_efficiency'] = abs(data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    )
    
    data['long_term_efficiency'] = abs(data['close'] - data['close'].shift(13)) / (
        data['high'].rolling(window=13).max() - data['low'].rolling(window=13).min()
    )
    
    # Multi-Timeframe Volume Anomaly
    data['volume_acceleration'] = (
        (data['volume'] / data['volume'].shift(2) - 1) - 
        (data['volume'] / data['volume'].shift(5) - 1)
    )
    
    # Volume persistence: count consistent volume direction changes
    def calc_volume_persistence(volume_series):
        volume_changes = volume_series.diff().fillna(0)
        signs = np.sign(volume_changes)
        window = 4
        persistence = []
        for i in range(len(volume_series)):
            if i < window:
                persistence.append(0)
                continue
            window_signs = signs.iloc[i-window+1:i+1]
            if len(window_signs) == 0:
                persistence.append(0)
                continue
            consistent_count = sum(window_signs == window_signs.iloc[0])
            persistence.append(consistent_count)
        return pd.Series(persistence, index=volume_series.index)
    
    data['volume_persistence'] = calc_volume_persistence(data['volume'])
    
    data['volume_breakout'] = data['volume'] / data['volume'].rolling(window=7).max()
    
    # Price-Volume Divergence Integration
    data['efficiency_volume_divergence'] = data['short_term_efficiency'] * data['volume_acceleration']
    
    # Volume-anchored efficiency
    def calc_volume_anchored_efficiency(data_window):
        if len(data_window) < 5:
            return 0
        volume_weighted_moves = sum(
            data_window['volume'].iloc[i] * abs(data_window['close'].iloc[i] - data_window['close'].iloc[i-1]) 
            for i in range(1, len(data_window))
        )
        if volume_weighted_moves == 0:
            return 0
        return (data_window['close'].iloc[-1] - data_window['close'].iloc[0]) / volume_weighted_moves
    
    data['volume_anchored_efficiency'] = data['close'].rolling(window=6).apply(
        lambda x: calc_volume_anchored_efficiency(data.loc[x.index]), raw=False
    )
    
    data['breakout_divergence'] = data['volume_breakout'] * data['long_term_efficiency']
    
    # Regime-Sensitive Momentum System
    # Momentum Regime Analysis
    def calc_momentum_regime_shift(close_series):
        close_changes = close_series.diff().fillna(0)
        signs = np.sign(close_changes)
        window = 6
        regime_shifts = []
        for i in range(len(close_series)):
            if i < window:
                regime_shifts.append(0)
                continue
            window_signs = signs.iloc[i-window+1:i+1]
            if len(window_signs) == 0:
                regime_shifts.append(0)
                continue
            shift_count = sum(window_signs.iloc[j] != window_signs.iloc[j-1] for j in range(1, len(window_signs)))
            regime_shifts.append(shift_count)
        return pd.Series(regime_shifts, index=close_series.index)
    
    data['momentum_regime_shift'] = calc_momentum_regime_shift(data['close'])
    
    data['regime_momentum'] = (data['close'] / data['close'].shift(3) - 1) * (1 - data['momentum_regime_shift'] / 5)
    
    data['regime_breakout'] = data['close'] / data['low'].rolling(window=9).min()
    
    # Volume-Regime Dynamics
    data['volume_regime_alignment'] = (
        np.sign(data['close'] - data['close'].shift(1)) * 
        np.sign(data['volume'] - data['volume'].rolling(window=3).mean().shift(1))
    )
    
    data['regime_volume_intensity'] = (
        data['volume'] / (data['high'] - data['low']) * abs(data['close'] - data['close'].shift(1))
    )
    
    # Volume regime persistence
    def calc_volume_regime_persistence(volume_series):
        volume_increases = volume_series > volume_series.shift(1)
        window = 3
        persistence = []
        for i in range(len(volume_series)):
            if i < window:
                persistence.append(0)
                continue
            window_increases = volume_increases.iloc[i-window+1:i+1]
            persistence.append(window_increases.sum())
        return pd.Series(persistence, index=volume_series.index)
    
    data['volume_regime_persistence'] = calc_volume_regime_persistence(data['volume'])
    
    # Regime-Momentum Integration
    data['regime_enhanced_momentum'] = data['regime_momentum'] * data['volume_regime_alignment']
    
    # Volume-momentum efficiency
    def calc_volume_momentum_efficiency(data_window):
        if len(data_window) < 6:
            return 0
        volume_weighted_moves = sum(
            data_window['volume'].iloc[i] * abs(data_window['close'].iloc[i] - data_window['close'].iloc[i-1]) 
            for i in range(1, len(data_window))
        )
        if volume_weighted_moves == 0:
            return 0
        persistence_factor = data_window['volume_regime_persistence'].iloc[-1] if data_window['volume_regime_persistence'].iloc[-1] > 0 else 1
        return (data_window['close'].iloc[-1] - data_window['close'].iloc[0]) / (volume_weighted_moves * persistence_factor)
    
    data['volume_momentum_efficiency'] = data['close'].rolling(window=6).apply(
        lambda x: calc_volume_momentum_efficiency(data.loc[x.index]), raw=False
    )
    
    data['regime_breakout_momentum'] = data['regime_breakout'] * data['regime_volume_intensity']
    
    # Divergence-Convergence Detection
    # Efficiency Divergence Analysis
    data['efficiency_convergence'] = data['short_term_efficiency'] / data['medium_term_efficiency']
    
    data['efficiency_divergence_momentum'] = (
        (data['short_term_efficiency'] - data['medium_term_efficiency']) - 
        (data['medium_term_efficiency'] - data['long_term_efficiency'])
    )
    
    # Efficiency regime stability
    def calc_efficiency_regime_stability(efficiency_series):
        efficiency_changes = efficiency_series.diff().abs()
        window = 3
        stability = []
        for i in range(len(efficiency_series)):
            if i < window:
                stability.append(0)
                continue
            window_changes = efficiency_changes.iloc[i-window+1:i+1]
            stable_count = sum(window_changes < 0.1)
            stability.append(stable_count)
        return pd.Series(stability, index=efficiency_series.index)
    
    data['efficiency_regime_stability'] = calc_efficiency_regime_stability(data['short_term_efficiency'])
    
    # Volume Divergence Analysis
    # Volume-price divergence (correlation)
    def calc_volume_price_correlation(window_data):
        if len(window_data) < 6:
            return 0
        volumes = window_data['volume'].values
        price_moves = abs(window_data['close'].diff().fillna(0)).values[1:]
        if len(volumes) != len(price_moves) + 1:
            return 0
        volumes = volumes[1:]  # Align with price moves
        if len(volumes) < 2:
            return 0
        try:
            correlation = np.corrcoef(volumes, price_moves)[0, 1]
            return correlation if not np.isnan(correlation) else 0
        except:
            return 0
    
    data['volume_price_divergence'] = data['close'].rolling(window=6).apply(
        lambda x: calc_volume_price_correlation(data.loc[x.index]), raw=False
    )
    
    data['divergence_strength'] = abs(data['volume_acceleration']) * data['volume_persistence']
    
    data['volume_convergence_signal'] = data['volume_breakout'] * data['volume_regime_alignment']
    
    # Divergence-Convergence Integration
    data['efficiency_volume_convergence'] = data['efficiency_convergence'] * data['volume_price_divergence']
    
    data['regime_divergence_alignment'] = data['volume_regime_alignment'] * data['efficiency_divergence_momentum']
    
    data['convergence_transition'] = data['efficiency_regime_stability'] * data['divergence_strength']
    
    # Adaptive Alpha Synthesis
    # Core Divergence Signals
    data['primary_divergence'] = data['efficiency_volume_divergence'] * data['volume_anchored_efficiency']
    
    data['regime_confirmation'] = data['regime_enhanced_momentum'] * data['volume_momentum_efficiency']
    
    data['convergence_enhancement'] = data['breakout_divergence'] * data['efficiency_volume_convergence']
    
    # Signal Adaptation
    data['base_alpha'] = data['primary_divergence'] * data['regime_confirmation']
    
    data['convergence_adjustment'] = data['base_alpha'] * (1 + data['convergence_transition'])
    
    data['regime_amplification'] = data['convergence_adjustment'] * data['regime_breakout_momentum']
    
    # Final Alpha Generation
    data['directional_refinement'] = data['regime_amplification'] * np.sign(data['primary_divergence'])
    
    data['volume_convergence_confirmation'] = data['directional_refinement'] * abs(data['regime_confirmation'])
    
    data['final_alpha'] = data['volume_convergence_confirmation'] * data['convergence_enhancement']
    
    # Return the final alpha factor
    return data['final_alpha']
