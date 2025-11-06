import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Entropy-Momentum Framework for alpha generation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(3, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Scale Momentum Integration
        # Volume-Adjusted Momentum
        if i >= 3:
            vol_adj_momentum = ((current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / 
                               current_data['close'].iloc[i-3]) * current_data['volume'].iloc[i]
        else:
            vol_adj_momentum = 0
        
        # Fractal Momentum Decay (simplified)
        if i >= 5:
            momentum_5d = (current_data['close'].iloc[i] - current_data['close'].iloc[i-5]) / current_data['close'].iloc[i-5]
            momentum_3d = (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / current_data['close'].iloc[i-3]
            momentum_decay = momentum_3d - momentum_5d if momentum_5d != 0 else 0
        else:
            momentum_decay = 0
        
        # Volume-Flow Synchronization
        # Entropy-Weighted Flow
        recent_volume = current_data['volume'].iloc[max(0, i-4):i+1]
        volume_entropy = -np.sum((recent_volume / recent_volume.sum()) * 
                                np.log(recent_volume / recent_volume.sum() + 1e-8))
        
        if i >= 1:
            flow_imbalance = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) * current_data['volume'].iloc[i]
            entropy_weighted_flow = flow_imbalance * (1 - volume_entropy / np.log(len(recent_volume)))
        else:
            entropy_weighted_flow = 0
        
        # Entropy Breakout Detection
        # Compression Breakout
        if i >= 10:
            price_range = current_data['high'].iloc[i-9:i+1].max() - current_data['low'].iloc[i-9:i+1].min()
            recent_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            breakout_score = recent_range / price_range if price_range > 0 else 0
            compression_breakout = breakout_score * (1 - volume_entropy / np.log(10))
        else:
            compression_breakout = 0
        
        # Volume Breakout
        if i >= 5:
            avg_volume = current_data['volume'].iloc[i-4:i+1].mean()
            volume_breakout = (current_data['volume'].iloc[i] - avg_volume) / avg_volume if avg_volume > 0 else 0
            volume_breakout_adj = volume_breakout * (1 - volume_entropy / np.log(5))
        else:
            volume_breakout_adj = 0
        
        # Efficiency-Entropy Patterns
        # Entropy-Adjusted Efficiency
        if i >= 3:
            true_range = max(
                current_data['high'].iloc[i] - current_data['low'].iloc[i],
                abs(current_data['high'].iloc[i] - current_data['close'].iloc[i-1]),
                abs(current_data['low'].iloc[i] - current_data['close'].iloc[i-1])
            )
            price_move = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1])
            directional_efficiency = price_move / true_range if true_range > 0 else 0
            entropy_adj_efficiency = directional_efficiency * (1 - volume_entropy / np.log(4))
        else:
            entropy_adj_efficiency = 0
        
        # Microstructure Divergence
        # Timeframe Alignment
        if i >= 8:
            short_term_data = current_data.iloc[i-2:i+1]
            medium_term_data = current_data.iloc[i-7:i+1]
            
            short_volume_entropy = -np.sum((short_term_data['volume'] / short_term_data['volume'].sum()) * 
                                         np.log(short_term_data['volume'] / short_term_data['volume'].sum() + 1e-8))
            medium_volume_entropy = -np.sum((medium_term_data['volume'] / medium_term_data['volume'].sum()) * 
                                          np.log(medium_term_data['volume'] / medium_term_data['volume'].sum() + 1e-8))
            
            timeframe_alignment = short_volume_entropy - medium_volume_entropy
        else:
            timeframe_alignment = 0
        
        # Composite Alpha Calculation
        composite_alpha = (
            vol_adj_momentum * 0.2 +
            momentum_decay * 0.15 +
            entropy_weighted_flow * 0.25 +
            compression_breakout * 0.1 +
            volume_breakout_adj * 0.1 +
            entropy_adj_efficiency * 0.1 +
            timeframe_alignment * 0.1
        )
        
        alpha.iloc[i] = composite_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
