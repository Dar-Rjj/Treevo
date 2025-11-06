import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components with proper shifting to avoid lookahead bias
    for i in range(2, len(df)):
        # Current day data
        open_t = df.iloc[i]['open']
        high_t = df.iloc[i]['high']
        low_t = df.iloc[i]['low']
        close_t = df.iloc[i]['close']
        amount_t = df.iloc[i]['amount']
        volume_t = df.iloc[i]['volume']
        
        # Historical data
        close_t1 = df.iloc[i-1]['close']
        close_t2 = df.iloc[i-2]['close']
        high_t1 = df.iloc[i-1]['high']
        low_t1 = df.iloc[i-1]['low']
        high_t2 = df.iloc[i-2]['high']
        low_t2 = df.iloc[i-2]['low']
        amount_t1 = df.iloc[i-1]['amount']
        volume_t1 = df.iloc[i-1]['volume']
        
        # Avoid division by zero
        epsilon = 1e-8
        
        # Regime-Dependent Momentum Components
        regime_adaptive_price_momentum = ((close_t / (close_t2 + epsilon) - 1) * 
                                        (high_t - low_t) / (max(high_t2 - low_t2, epsilon)))
        
        bidirectional_flow_momentum = ((close_t - open_t) * 
                                     (amount_t - amount_t1) / max(volume_t, epsilon))
        
        volatility_weighted_return = ((close_t / (close_t1 + epsilon) - 1) * 
                                    (high_t - low_t) / max(abs(close_t - open_t), epsilon))
        
        # Flow-Direction Microstructure Analysis
        bidirectional_pressure_ratio = (abs(close_t - open_t) / max(high_t - low_t, epsilon) * 
                                      volume_t)
        
        flow_direction_confirmation = ((close_t - (high_t + low_t)/2) * 
                                     (volume_t - volume_t1) / max(amount_t, epsilon))
        
        microstructure_flow_divergence = (abs((high_t + low_t)/2 - close_t) * 
                                        abs(volume_t - volume_t1) / max(high_t - low_t, epsilon))
        
        # Volatility-Regime Transition Dynamics
        regime_transition_signal = ((high_t - low_t) / max(high_t1 - low_t1, epsilon) * 
                                  (volume_t / max(volume_t1, epsilon) - 1))
        
        volatility_persistence_ratio = ((high_t - low_t) / 
                                      max((high_t1 - low_t1) + (high_t2 - low_t2)/2, epsilon))
        
        flow_regime_correlation = ((close_t - open_t) * (volume_t - volume_t1) / 
                                 max((high_t - low_t) * amount_t, epsilon))
        
        # Adaptive Momentum Synthesis
        regime_adaptive_core_momentum = (regime_adaptive_price_momentum * 
                                       volatility_weighted_return)
        
        flow_confirmed_momentum = (bidirectional_flow_momentum * 
                                 flow_direction_confirmation)
        
        volatility_flow_integration = (volatility_persistence_ratio * 
                                     flow_regime_correlation)
        
        # Bidirectional Microstructure Integration
        pressure_flow_wave = (((close_t - low_t) / max(high_t - low_t, epsilon)) * 
                            (amount_t / max(volume_t, epsilon) - 1))
        
        directional_flow_interference = ((close_t - open_t) * 
                                       (volume_t / max(volume_t1, epsilon) - 1) * 
                                       (amount_t / max(amount_t1, epsilon) - 1))
        
        microstructure_position_flow = (abs((high_t + low_t)/2 - close_t) * 
                                      (volume_t - volume_t1) * 
                                      (close_t / max(close_t1, epsilon) - 1))
        
        # Composite Adaptive Alpha Synthesis
        core_adaptive_signal = (regime_adaptive_core_momentum * 
                              flow_confirmed_momentum)
        
        volatility_flow_dynamics = (volatility_flow_integration * 
                                  regime_transition_signal)
        
        microstructure_confirmation = (bidirectional_pressure_ratio * 
                                     microstructure_flow_divergence)
        
        # Final Alpha Factor
        final_alpha = (core_adaptive_signal * volatility_flow_dynamics * 
                      microstructure_confirmation * directional_flow_interference)
        
        alpha.iloc[i] = final_alpha
    
    # Fill NaN values with 0 for the first two days
    alpha = alpha.fillna(0)
    
    return alpha
