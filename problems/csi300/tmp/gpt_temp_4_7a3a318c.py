import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility Fractal Flow with Adaptive Reversal alpha factor
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Fractal Microstructure Pressure Analysis
    # Net Session Pressure
    high_low_range = data['high'] - data['low'] + epsilon
    net_session_pressure = ((data['high'] - data['close']) - (data['close'] - data['low'])) / high_low_range * data['volume']
    
    # Session Fractal Divergence
    session_fractal_divergence = (
        (data['high'] - data['open']) / (data['open'] - data['low'] + epsilon) - 
        (data['close'] - data['low']) / (data['high'] - data['close'] + epsilon)
    ) * data['volume']
    
    # Combined Pressure Signal
    combined_pressure_signal = net_session_pressure * session_fractal_divergence
    
    # Multi-Timeframe Flow-Reversal Synthesis
    # Bidirectional Flow Imbalance
    upside_flow = pd.Series(0.0, index=data.index)
    downside_flow = pd.Series(0.0, index=data.index)
    
    for i in range(5):
        close_shifted = data['close'].shift(i)
        open_shifted = data['open'].shift(i)
        volume_shifted = data['volume'].shift(i)
        
        # Upside flow (close > open)
        upside_mask = close_shifted > open_shifted
        upside_flow += upside_mask * (close_shifted - open_shifted) * volume_shifted
        
        # Downside flow (close < open)
        downside_mask = close_shifted < open_shifted
        downside_flow += downside_mask * (open_shifted - close_shifted) * volume_shifted
    
    # Net Flow Imbalance
    net_flow_imbalance = np.log(upside_flow / (downside_flow + epsilon))
    
    # Multi-Timeframe Reversal
    # Short-term Reversal
    close_std_short = data['close'].rolling(window=5, min_periods=1).std() + epsilon
    short_term_reversal = (
        np.sign(data['close'].shift(2) - data['close']) * 
        np.abs((data['close'].shift(2) - data['close']) / close_std_short)
    )
    
    # Medium-term Reversal
    close_std_medium = data['close'].rolling(window=11, min_periods=1).std() + epsilon
    medium_term_reversal = (
        np.sign(data['close'].shift(5) - data['close']) * 
        np.abs((data['close'].shift(5) - data['close']) / close_std_medium)
    )
    
    # Combined Reversal
    combined_reversal = np.minimum(short_term_reversal, medium_term_reversal)
    
    # Flow-Enhanced Reversal
    flow_enhanced_reversal = net_flow_imbalance * combined_reversal
    
    # Volatility Fractal Regime Adaptation
    # Volatility Scaling Analysis
    short_term_vol = (data['high'] - data['low']).rolling(window=2, min_periods=1).mean()
    medium_term_vol = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    volatility_scaling_ratio = short_term_vol / (medium_term_vol + epsilon)
    
    # Regime Classification
    vol_5day = (data['high'] - data['low']).rolling(window=5, min_periods=1).mean()
    vol_10day = (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    
    high_vol_regime = (vol_5day > vol_10day).astype(float)
    low_vol_regime = (vol_5day < vol_10day).astype(float)
    
    # Regime Transition
    regime_transition = high_vol_regime.diff().abs()
    
    # Adaptive Multipliers
    high_vol_multiplier = 1 + volatility_scaling_ratio * high_vol_regime
    low_vol_multiplier = 1 + (1 / (volatility_scaling_ratio + epsilon)) * low_vol_regime
    transition_boost = 1.5 * regime_transition
    
    # Combined Adaptive Multiplier
    adaptive_multipliers = high_vol_multiplier + low_vol_multiplier - 1 + transition_boost
    
    # Fractal Flow Persistence Confirmation
    # Flow Direction Persistence
    net_flow_sign = np.sign(net_flow_imbalance)
    consecutive_flow_days = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        if net_flow_sign.iloc[i] == net_flow_sign.iloc[i-1]:
            consecutive_flow_days.iloc[i] = consecutive_flow_days.iloc[i-1] + 1
        else:
            consecutive_flow_days.iloc[i] = 1
    
    persistence_strength = np.minimum(consecutive_flow_days, 5)
    
    # Fractal Consistency
    fractal_patterns = []
    for i in range(len(data)):
        if i >= 5:
            recent_pattern = session_fractal_divergence.iloc[i-2:i+1].values
            past_pattern = session_fractal_divergence.iloc[i-5:i-2].values
            if len(recent_pattern) == 3 and len(past_pattern) == 3:
                corr = np.corrcoef(recent_pattern, past_pattern)[0, 1]
                if not np.isnan(corr):
                    fractal_patterns.append(corr)
                else:
                    fractal_patterns.append(0)
            else:
                fractal_patterns.append(0)
        else:
            fractal_patterns.append(0)
    
    fractal_stability = pd.Series(fractal_patterns, index=data.index).abs()
    
    # Confirmation Signal
    confirmation_signal = persistence_strength * fractal_stability
    
    # Integrated Alpha Generation
    # Core Fractal Flow
    core_fractal_flow = combined_pressure_signal * flow_enhanced_reversal
    
    # Regime-Adapted Signal
    regime_adapted_signal = core_fractal_flow * adaptive_multipliers
    
    # Persistence-Filtered Signal
    persistence_filtered_signal = regime_adapted_signal * confirmation_signal
    
    # Final Alpha Factor
    final_alpha = np.sign(persistence_filtered_signal) * np.abs(persistence_filtered_signal) ** (1/3)
    
    return final_alpha
