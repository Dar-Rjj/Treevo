import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need enough history for calculations
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        # Historical data points
        t_1 = df.iloc[i-1] if i >= 1 else None
        t_2 = df.iloc[i-2] if i >= 2 else None
        t_5 = df.iloc[i-5] if i >= 5 else None
        t_10 = df.iloc[i-10] if i >= 10 else None
        t_11 = df.iloc[i-11] if i >= 11 else None
        t_20 = df.iloc[i-20] if i >= 20 else None
        
        # Directional Absorption Analysis
        upward_absorption = (current_data['high'] - current_data['open']) / (current_data['volume'] + 1e-8)
        downward_resistance = (current_data['open'] - current_data['low']) / (current_data['volume'] + 1e-8)
        net_absorption_flow = (upward_absorption - downward_resistance) / (upward_absorption + downward_resistance + 1e-8)
        
        # Cross-Timeframe Momentum Decoupling
        if t_5 is not None and t_20 is not None:
            short_long_divergence = ((current_data['close'] - t_5['close']) / (t_5['close'] - t_20['close'] + 1e-8)) * current_data['volume']
        else:
            short_long_divergence = 0
            
        if t_5 is not None and t_20 is not None:
            volatility_breakdown = ((current_data['high'] - current_data['low']) / (t_5['high'] - t_5['low'] + 1e-8)) * ((t_20['high'] - t_20['low']) / (t_5['high'] - t_5['low'] + 1e-8))
        else:
            volatility_breakdown = 0
            
        if t_5 is not None and t_20 is not None:
            volume_correlation = (current_data['volume'] / (t_5['volume'] + 1e-8)) * (t_5['volume'] / (t_20['volume'] + 1e-8))
        else:
            volume_correlation = 0
        
        # Memory-Based Absorption Dynamics
        if t_1 is not None and t_10 is not None and t_11 is not None:
            price_persistence = ((current_data['close'] - t_1['close']) / (t_10['close'] - t_11['close'] + 1e-8)) * (current_data['volume'] / (t_10['volume'] + 1e-8))
        else:
            price_persistence = 0
            
        if t_10 is not None:
            volatility_decay = ((current_data['high'] - current_data['low']) / (t_10['high'] - t_10['low'] + 1e-8)) * (current_data['volume'] / (t_10['volume'] + 1e-8))
        else:
            volatility_decay = 0
            
        if t_1 is not None and t_10 is not None and t_11 is not None:
            gap_memory = (abs(current_data['open'] - t_1['close']) / (abs(t_10['open'] - t_11['close']) + 1e-8)) * (current_data['volume'] / (t_10['volume'] + 1e-8))
        else:
            gap_memory = 0
        
        # Absorption Microstructure
        large_order_efficiency = current_data['amount'] / (current_data['high'] - current_data['low'] + 1e-8)
        
        if t_2 is not None:
            volume_avg = (df.iloc[i-2:i+1]['volume'].mean() if i >= 2 else current_data['volume'])
            order_concentration = current_data['volume'] / (volume_avg + 1e-8)
        else:
            order_concentration = 1
            
        price_absorption_rate = abs(current_data['close'] - current_data['open']) / (current_data['high'] - current_data['low'] + 1e-8)
        
        # Regime Classification
        if price_absorption_rate > 0.7:
            absorption_regime = 'Strong'
        elif price_absorption_rate < 0.3:
            absorption_regime = 'Weak'
        else:
            absorption_regime = 'Neutral'
            
        if short_long_divergence > 1:
            momentum_regime = 'Accelerating'
        elif short_long_divergence < -1:
            momentum_regime = 'Decelerating'
        else:
            momentum_regime = 'Stable'
            
        if price_persistence > 1:
            memory_regime = 'High Memory'
        elif price_persistence < 0.5:
            memory_regime = 'Low Memory'
        else:
            memory_regime = 'Normal Memory'
        
        # Cross-Regime Alpha Construction
        if absorption_regime == 'Strong' and momentum_regime == 'Accelerating':
            factor = net_absorption_flow * short_long_divergence * price_absorption_rate
        elif absorption_regime == 'Weak' and memory_regime == 'High Memory':
            factor = gap_memory * large_order_efficiency * price_persistence
        elif absorption_regime == 'Neutral' and momentum_regime == 'Decelerating':
            factor = volatility_breakdown * volume_correlation * order_concentration
        else:
            # Mixed regime weighted average
            absorption_strength = abs(price_absorption_rate - 0.5)
            momentum_strength = abs(short_long_divergence)
            memory_strength = abs(price_persistence - 0.75)
            
            total_strength = absorption_strength + momentum_strength + memory_strength + 1e-8
            
            factor = (absorption_strength * net_absorption_flow +
                     momentum_strength * short_long_divergence +
                     memory_strength * price_persistence) / total_strength
        
        alpha.iloc[i] = factor
    
    # Temporal smoothing
    smoothed_alpha = alpha.copy()
    for i in range(2, len(alpha)):
        if not pd.isna(alpha.iloc[i]) and not pd.isna(alpha.iloc[i-1]) and not pd.isna(alpha.iloc[i-2]):
            smoothed_alpha.iloc[i] = (alpha.iloc[i] + alpha.iloc[i-1] + alpha.iloc[i-2]) / 3
    
    return smoothed_alpha
