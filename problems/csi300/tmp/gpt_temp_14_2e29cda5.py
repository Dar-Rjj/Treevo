import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Asymmetric Gap Dynamics with Rejection Absorption alpha factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(2, len(df)):
        if i < 5:  # Need at least 5 days for some calculations
            alpha.iloc[i] = 0
            continue
            
        # Current day data
        open_t = df['open'].iloc[i]
        high_t = df['high'].iloc[i]
        low_t = df['low'].iloc[i]
        close_t = df['close'].iloc[i]
        volume_t = df['volume'].iloc[i]
        amount_t = df['amount'].iloc[i]
        
        # Previous day data
        close_t1 = df['close'].iloc[i-1]
        amount_t1 = df['amount'].iloc[i-1]
        amount_t2 = df['amount'].iloc[i-2] if i >= 2 else amount_t1
        
        # 1. Asymmetric Gap Rejection Analysis
        # Overnight Gap Rejection
        gap_magnitude = abs(open_t - close_t1)
        daily_range = max(open_t, close_t) - min(open_t, close_t)
        gap_rejection_magnitude = gap_magnitude - daily_range
        
        gap_absorption_efficiency = abs(close_t - open_t) / (gap_magnitude + 1e-8)
        gap_direction_persistence = np.sign(open_t - close_t1) * np.sign(close_t - open_t)
        
        # Intraday Rejection Asymmetry
        upper_rejection_intensity = (high_t - max(open_t, close_t)) / (high_t - low_t + 1e-8)
        lower_rejection_intensity = (min(open_t, close_t) - low_t) / (high_t - low_t + 1e-8)
        net_rejection_asymmetry = upper_rejection_intensity - lower_rejection_intensity
        
        # Multi-period Gap Memory
        gap_pressure_3d = 0
        for j in range(3):
            if i-j >= 1:
                open_j = df['open'].iloc[i-j]
                close_j1 = df['close'].iloc[i-j-1] if i-j-1 >= 0 else open_j
                close_j = df['close'].iloc[i-j]
                gap_pressure_3d += abs(open_j - close_j1) - abs(close_j - open_j)
        
        # Gap clustering persistence
        gap_directions = []
        for j in range(5):
            if i-j >= 1:
                open_j = df['open'].iloc[i-j]
                close_j1 = df['close'].iloc[i-j-1] if i-j-1 >= 0 else open_j
                gap_directions.append(np.sign(open_j - close_j1))
        
        gap_clustering_persistence = 0
        if len(gap_directions) >= 2:
            for j in range(1, len(gap_directions)):
                if gap_directions[j] == gap_directions[j-1] and gap_directions[j] != 0:
                    gap_clustering_persistence += 1
        
        # 2. Liquidity Rejection Absorption Patterns
        # Volume Rejection Distribution
        high_side_absorption_volume = volume_t * upper_rejection_intensity
        low_side_absorption_volume = volume_t * lower_rejection_intensity
        net_absorption_asymmetry = high_side_absorption_volume - low_side_absorption_volume
        
        # Amount Flow Rejection Dynamics
        amount_rejection_velocity = (amount_t / (amount_t1 + 1e-8)) - 1
        amount_rejection_persistence = (np.sign(amount_t - amount_t1) * 
                                      np.sign(amount_t1 - amount_t2))
        amount_rejection_efficiency = (abs(close_t - open_t) * amount_t / 
                                     (high_t - low_t + 1e-8))
        
        # Multi-scale Absorption Efficiency
        mid_price = (high_t + low_t) / 2
        opening_session_absorption = abs(mid_price - open_t) / (high_t - low_t + 1e-8)
        closing_session_absorption = abs(close_t - mid_price) / (high_t - low_t + 1e-8)
        full_session_directional_absorption = (np.sign(close_t - open_t) * 
                                             np.sign(mid_price - open_t))
        
        # 3. Rejection-Gap Phase Analysis
        # Gap direction vs rejection asymmetry alignment
        gap_rejection_alignment = np.sign(open_t - close_t1) * net_rejection_asymmetry
        
        # Multi-timeframe Rejection Dynamics
        # 3-day rejection acceleration
        close_3d_max = max(df['close'].iloc[i-2:i+1])
        close_3d_min = min(df['close'].iloc[i-2:i+1])
        rejection_acceleration_3d = ((high_t - close_3d_max) - 
                                   (close_3d_min - low_t))
        
        # 10-day rejection persistence (using available data)
        lookback = min(10, i)
        close_10d_max = max(df['close'].iloc[i-lookback+1:i+1])
        close_10d_min = min(df['close'].iloc[i-lookback+1:i+1])
        rejection_persistence_10d = ((high_t - close_10d_max) - 
                                   (close_10d_min - low_t))
        
        # 4. Dynamic Alpha Factor Construction
        # Asymmetric Gap Rejection Score
        gap_rejection_score = (gap_rejection_magnitude * 0.3 + 
                              net_rejection_asymmetry * 0.4 + 
                              gap_absorption_efficiency * 0.3)
        
        # Liquidity Absorption Multiplier
        liquidity_multiplier = (net_absorption_asymmetry * 0.4 + 
                               amount_rejection_efficiency * 0.3 + 
                               (opening_session_absorption + closing_session_absorption) * 0.3)
        
        # Multi-timeframe dynamics
        timeframe_dynamics = (rejection_acceleration_3d * 0.6 + 
                            rejection_persistence_10d * 0.4)
        
        # Final composite alpha factor
        final_alpha = (gap_rejection_score * 0.4 + 
                      liquidity_multiplier * 0.3 + 
                      timeframe_dynamics * 0.2 + 
                      gap_clustering_persistence * 0.1)
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial values
    alpha = alpha.fillna(0)
    
    return alpha
