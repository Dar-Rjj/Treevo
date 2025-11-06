import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Asset Quantum Volatility Microstructure Alpha Factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(34, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Quantum Volatility Momentum Framework
        # Micro-Quantum Momentum (5-day)
        if i >= 5:
            micro_momentum = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * 
                            current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                            (current_data['high'].iloc[i-1:i+1].max() - current_data['low'].iloc[i-1:i+1].min()))
        else:
            micro_momentum = 0
        
        # Meso-Quantum Momentum (13-day)
        if i >= 13:
            meso_momentum = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) * 
                           current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] / 
                           (current_data['high'].iloc[i-5:i+1].max() - current_data['low'].iloc[i-5:i+1].min()))
        else:
            meso_momentum = 0
        
        # Macro-Quantum Momentum (34-day)
        if i >= 34:
            macro_momentum = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-13]) * 
                            current_data['volume'].iloc[i] / current_data['volume'].iloc[i-13] / 
                            (current_data['high'].iloc[i-13:i+1].max() - current_data['low'].iloc[i-13:i+1].min()))
        else:
            macro_momentum = 0
        
        # Quantum Volume-Volatility Entanglement
        if i >= 2:
            # Quantum Volume Efficiency
            q_volume_eff = ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) * 
                          current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                          (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
            
            # Quantum Volatility Absorption
            q_vol_absorption = (abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) * 
                              current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                              (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
            
            # Quantum Volume Exhaustion
            q_volume_exhaustion = (((current_data['close'].iloc[i] - current_data['open'].iloc[i]) * 
                                  current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] - 
                                  (current_data['close'].iloc[i-1] - current_data['open'].iloc[i-1]) * 
                                  current_data['volume'].iloc[i-1] / current_data['volume'].iloc[i-2]) / 
                                 (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        else:
            q_volume_eff = q_vol_absorption = q_volume_exhaustion = 0
        
        # Quantum Price Efficiency Dynamics
        if i >= 2:
            # Quantum Efficiency Asymmetry
            q_efficiency_asym = (q_volume_eff * current_data['amount'].iloc[i] / current_data['volume'].iloc[i] / 
                               (current_data['amount'].iloc[i-1] / current_data['volume'].iloc[i-1]))
            
            # Quantum Volatility Momentum
            q_vol_momentum = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) * 
                            current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                            (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]))
            
            # Quantum Position Efficiency
            q_position_eff = ((current_data['close'].iloc[i] - current_data['low'].iloc[i]) * 
                            current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                            (current_data['high'].iloc[i] - current_data['low'].iloc[i]))
        else:
            q_efficiency_asym = q_vol_momentum = q_position_eff = 0
        
        # Quantum Entanglement Persistence
        if i >= 4:
            # Short-term Quantum Consistency (3-day)
            if i >= 6:
                short_consistency = (np.sign(q_volume_eff) * np.sign(q_volume_eff) * np.sign(q_volume_eff))
            else:
                short_consistency = 1
            
            # Medium-term Quantum Consistency (5-day)
            if i >= 8:
                medium_consistency = (np.sign(q_vol_absorption) * np.sign(q_vol_absorption) * 
                                    np.sign(q_vol_absorption) * np.sign(q_vol_absorption) * np.sign(q_vol_absorption))
            else:
                medium_consistency = 1
            
            # Quantum Momentum-to-Volatility Ratio
            q_momentum_vol_ratio = macro_momentum / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0 else 0
        else:
            short_consistency = medium_consistency = q_momentum_vol_ratio = 0
        
        # Multi-Dimensional Quantum Integration
        if i >= 1:
            # Intraday Quantum Entanglement
            intraday_entanglement = ((current_data['high'].iloc[i] - current_data['low'].iloc[i]) * 
                                   current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                                   (current_data['close'].iloc[i] - current_data['open'].iloc[i]) * 
                                   current_data['volume'].iloc[i] / current_data['amount'].iloc[i]) if abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) > 0 else 0
            
            # Overnight Quantum Gap
            overnight_gap = ((current_data['open'].iloc[i] - current_data['close'].iloc[i-1]) * 
                           current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                           (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1])) if (current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]) > 0 else 0
            
            # Quantum Pressure Dynamics
            q_pressure = ((current_data['high'].iloc[i] - current_data['close'].iloc[i]) * 
                        (current_data['close'].iloc[i] - current_data['low'].iloc[i]) * 
                        current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] / 
                        (current_data['high'].iloc[i] - current_data['low'].iloc[i])) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) > 0 else 0
        else:
            intraday_entanglement = overnight_gap = q_pressure = 0
        
        # Hierarchical Quantum Signal Construction
        # Core Quantum Integration
        multi_scale_momentum = (micro_momentum * 0.4 + meso_momentum * 0.35 + macro_momentum * 0.25)
        quantum_volume_efficiency = (q_volume_eff * 0.4 + q_vol_absorption * 0.3 + q_volume_exhaustion * 0.3)
        quantum_position_dynamics = (q_efficiency_asym * 0.3 + q_vol_momentum * 0.4 + q_position_eff * 0.3)
        
        core_integration = (multi_scale_momentum + quantum_volume_efficiency + quantum_position_dynamics) / 3
        
        # Quantum Persistence Enhancement
        persistence_weighted = core_integration * short_consistency * 0.3
        persistence_validated = persistence_weighted * medium_consistency * 0.35
        volatility_adjusted = persistence_validated * q_momentum_vol_ratio * 0.35
        
        # Final composite signal
        composite_signal = (core_integration + persistence_weighted + persistence_validated + volatility_adjusted) / 4
        
        # Normalize to quantum microstructure alpha ranges
        if composite_signal > 0.7:
            quantum_alpha = 0.85  # Strong Quantum Breakout
        elif composite_signal > 0.3:
            quantum_alpha = 0.5   # Moderate Quantum Trend
        elif composite_signal > 0.1:
            quantum_alpha = 0.2   # Weak Quantum Accumulation
        elif composite_signal > -0.1:
            quantum_alpha = 0     # Neutral Quantum
        elif composite_signal > -0.3:
            quantum_alpha = -0.2  # Weak Quantum Distribution
        elif composite_signal > -0.7:
            quantum_alpha = -0.5  # Moderate Quantum Reversal
        else:
            quantum_alpha = -0.85 # Strong Quantum Exhaustion
        
        result.iloc[i] = quantum_alpha
    
    # Fill early values with neutral signal
    result = result.fillna(0)
    
    return result
