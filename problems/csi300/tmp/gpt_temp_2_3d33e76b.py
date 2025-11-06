import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Coherence Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(5, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Volatility Regime Classification
        # Historical volatility patterns
        if i >= 5:
            # Realized volatility (5-day)
            realized_vol = np.sqrt(np.sum(np.square(np.diff(current_data['close'].iloc[-6:-1]))))
            # Range-based volatility
            range_vol = (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) / abs(current_data['close'].iloc[-2])
            # Volatility persistence
            vol_persistence = np.sign(
                (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) - 
                (current_data['high'].iloc[-2] - current_data['low'].iloc[-2])
            )
        else:
            realized_vol = range_vol = vol_persistence = 0
        
        # Volume-volatility relationship
        if i >= 1:
            vol_vol_coherence = current_data['volume'].iloc[-1] / max(1e-6, current_data['high'].iloc[-1] - current_data['low'].iloc[-1])
            
            # Abnormal volume impact
            price_change_threshold = 0.02 * current_data['close'].iloc[-2]
            if abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) > price_change_threshold:
                abnormal_vol_impact = current_data['volume'].iloc[-1] / max(1, current_data['volume'].iloc[-2])
            else:
                abnormal_vol_impact = 1
            
            # Volume spike regime
            volume_spike = 1 if current_data['volume'].iloc[-1] > 1.5 * current_data['volume'].iloc[-2] else 0
        else:
            vol_vol_coherence = abnormal_vol_impact = volume_spike = 0
        
        # Multi-timeframe volatility structure
        if i >= 10:
            # Short-term vs long-term volatility
            short_term_vol = np.std(np.diff(current_data['close'].iloc[-6:-1]))
            long_term_vol = np.std(np.diff(current_data['close'].iloc[-11:-1]))
            vol_ratio = short_term_vol / max(1e-6, long_term_vol)
            
            # Volatility acceleration
            vol_acceleration = (
                (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) - 
                (current_data['high'].iloc[-2] - current_data['low'].iloc[-2])
            )
        else:
            vol_ratio = vol_acceleration = 0
        
        # Price-Volume Coherence Dynamics
        if i >= 1:
            # Directional coherence measures
            price_sign = np.sign(current_data['close'].iloc[-1] - current_data['open'].iloc[-1])
            volume_sign = np.sign(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2])
            price_volume_alignment = price_sign * volume_sign
            
            # Coherence strength
            coherence_strength = (
                abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) * 
                abs(current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2])
            )
            
            # Divergence detection
            divergence = 1 if price_sign != volume_sign else 0
            
            # Relative movement coherence
            price_change_rel = (current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / max(1e-6, abs(current_data['close'].iloc[-2]))
            volume_change_rel = (current_data['volume'].iloc[-1] - current_data['volume'].iloc[-2]) / max(1, current_data['volume'].iloc[-2])
            relative_coherence = price_change_rel * volume_change_rel
        else:
            price_volume_alignment = coherence_strength = divergence = relative_coherence = 0
        
        # Regime-Adaptive Coherence Signals
        # Determine volatility regime
        if realized_vol > np.percentile([realized_vol], 70) if i >= 10 else realized_vol > 0.02:
            volatility_regime = 'high'
        elif realized_vol < np.percentile([realized_vol], 30) if i >= 10 else realized_vol < 0.005:
            volatility_regime = 'low'
        else:
            volatility_regime = 'medium'
        
        # High volatility regime signals
        if volatility_regime == 'high':
            volatility_enhanced_coherence = price_volume_alignment * range_vol
            spike_driven_signal = volume_spike * price_volume_alignment
        else:
            volatility_enhanced_coherence = spike_driven_signal = 0
        
        # Low volatility regime signals
        if volatility_regime == 'low':
            subtle_coherence = price_volume_alignment if abs(price_change_rel) < 0.01 else 0
            low_vol_breakout = coherence_strength / max(1e-6, range_vol)
        else:
            subtle_coherence = low_vol_breakout = 0
        
        # Core coherence factor construction
        # Primary coherence signal with volatility context
        primary_coherence = price_volume_alignment * (1 + range_vol)
        
        # Volatility regime adjustment
        if volatility_regime == 'high':
            regime_adjustment = 1.2
        elif volatility_regime == 'low':
            regime_adjustment = 0.8
        else:
            regime_adjustment = 1.0
        
        # Temporal enhancement (simple persistence)
        if i >= 3:
            recent_coherence = [
                np.sign(current_data['close'].iloc[-j] - current_data['open'].iloc[-j]) * 
                np.sign(current_data['volume'].iloc[-j] - current_data['volume'].iloc[-j-1])
                for j in range(2, min(4, i+1))
            ]
            coherence_persistence = sum([1 for x in recent_coherence if x > 0]) / len(recent_coherence)
        else:
            coherence_persistence = 0.5
        
        # Final alpha calculation
        core_factor = (
            primary_coherence * 
            regime_adjustment * 
            (1 + coherence_persistence) *
            (1 + relative_coherence)
        )
        
        # Risk management adjustment
        if i >= 10:
            recent_volatility = np.std(np.diff(current_data['close'].iloc[-10:]))
            risk_adjustment = 1 / max(0.01, recent_volatility)
        else:
            risk_adjustment = 1
        
        alpha.iloc[i] = core_factor * risk_adjustment
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
