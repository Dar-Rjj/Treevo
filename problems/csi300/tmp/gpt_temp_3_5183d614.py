import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical approaches with volume confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Ensure we have enough data for calculations
    min_periods = max(10, 8)  # Based on the longest lookback in the tree
    
    for i in range(min_periods, len(data)):
        current_idx = data.index[i]
        
        try:
            # Volatility-Adaptive Velocity Divergence
            # Velocity Components
            short_vel = (data['close'].iloc[i-1] - data['close'].iloc[i-3]) / data['close'].iloc[i-3]
            medium_vel = (data['close'].iloc[i-1] - data['close'].iloc[i-8]) / data['close'].iloc[i-8]
            vel_divergence = short_vel - medium_vel
            
            # Volatility Context
            tr_vol = max(
                data['high'].iloc[i-1] - data['low'].iloc[i-1],
                abs(data['high'].iloc[i-1] - data['close'].iloc[i-2]),
                abs(data['low'].iloc[i-1] - data['close'].iloc[i-2])
            )
            vol_compression = (data['high'].iloc[i-1] - data['low'].iloc[i-1]) / (data['high'].iloc[i-8] - data['low'].iloc[i-8])
            vol_adjustment = vel_divergence * vol_compression / (tr_vol + 1e-8)
            
            # Volume Confirmation
            vol_momentum = data['volume'].iloc[i-1] / (data['volume'].iloc[i-3] + 1e-8)
            vol_trend_consistency = np.sign(data['volume'].iloc[i-1] - data['volume'].iloc[i-5]) * np.sign(data['close'].iloc[i-1] - data['close'].iloc[i-5])
            factor1 = vol_adjustment * vol_momentum * vol_trend_consistency
            
            # Efficiency-Weighted Gap Reversal with Velocity
            # Gap Analysis
            opening_reversal = (data['close'].iloc[i-1] - data['open'].iloc[i]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            gap_persistence = (data['open'].iloc[i] - data['close'].iloc[i-2]) / (data['open'].iloc[i-1] - data['close'].iloc[i-3] + 1e-8)
            gap_strength = opening_reversal * gap_persistence
            
            # Efficiency Context
            range_efficiency = abs(data['close'].iloc[i-1] - data['close'].iloc[i-2]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            vel_confirmation = (data['close'].iloc[i-1] - data['close'].iloc[i-3]) / (data['close'].iloc[i-3] + 1e-8)
            efficiency_weight = gap_strength * range_efficiency * vel_confirmation
            
            # Volume Dynamics
            vol_reversal = (data['volume'].iloc[i] - data['volume'].iloc[i-2]) * (data['volume'].iloc[i-2] - data['volume'].iloc[i-4])
            vol_intensity = data['volume'].iloc[i-1] / (data['volume'].iloc[i-2] + data['volume'].iloc[i-3] + data['volume'].iloc[i-4] + 1e-8)
            factor2 = efficiency_weight * vol_reversal * vol_intensity
            
            # Range-Compression Breakthrough with Velocity Alignment
            # Compression Signals
            price_compression = (data['high'].iloc[i-1] - data['low'].iloc[i-1]) / (data['high'].iloc[i-10] - data['low'].iloc[i-10] + 1e-8)
            range_position = (data['close'].iloc[i-1] - data['low'].iloc[i-1]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            compression_intensity = price_compression * range_position
            
            # Breakthrough Detection
            vel_breakthrough = (data['close'].iloc[i-1] - data['close'].iloc[i-3]) / (data['close'].iloc[i-3] + 1e-8)
            vol_breakthrough = data['volume'].iloc[i] / (data['volume'].iloc[i-1] + 1e-8)
            breakthrough_signal = vel_breakthrough * vol_breakthrough
            
            # Efficiency Context
            daily_efficiency = abs(data['close'].iloc[i-1] - data['close'].iloc[i-2]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            efficiency_persistence = daily_efficiency / (abs(data['close'].iloc[i-2] - data['close'].iloc[i-3]) / (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 1e-8) + 1e-8)
            factor3 = compression_intensity * breakthrough_signal * daily_efficiency * efficiency_persistence
            
            # Multi-Timeframe Velocity-Efficiency Divergence
            # Velocity Components (reusing from first factor)
            vel_divergence2 = short_vel - medium_vel
            
            # Efficiency Context
            range_efficiency2 = abs(data['close'].iloc[i-1] - data['close'].iloc[i-2]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            efficiency_divergence = range_efficiency2 - (abs(data['close'].iloc[i-2] - data['close'].iloc[i-3]) / (data['high'].iloc[i-2] - data['low'].iloc[i-2] + 1e-8))
            efficiency_weight2 = vel_divergence2 * efficiency_divergence
            
            # Volume Confirmation
            vol_momentum2 = data['volume'].iloc[i-1] / (data['volume'].iloc[i-3] + 1e-8)
            vol_ratio = data['volume'].iloc[i-1] / ((data['volume'].iloc[i-2] + data['volume'].iloc[i-3] + data['volume'].iloc[i-4]) / 3 + 1e-8)
            factor4 = efficiency_weight2 * vol_momentum2 * vol_ratio
            
            # Dynamic Range-Velocity Alignment
            # Range Analysis
            position_ratio = (data['close'].iloc[i-1] - data['low'].iloc[i-1]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            avg_range = (data['high'].iloc[i-2] - data['low'].iloc[i-2] + data['high'].iloc[i-3] - data['low'].iloc[i-3] + data['high'].iloc[i-4] - data['low'].iloc[i-4]) / 3
            range_compression = (data['high'].iloc[i-1] - data['low'].iloc[i-1]) / (avg_range + 1e-8)
            range_quality = position_ratio * range_compression
            
            # Velocity Dynamics
            short_vel2 = (data['close'].iloc[i-1] - data['close'].iloc[i-3]) / (data['close'].iloc[i-3] + 1e-8)
            vel_consistency = (data['close'].iloc[i-1] / data['close'].iloc[i-3] - 1) / (data['close'].iloc[i-3] / data['close'].iloc[i-6] - 1 + 1e-8)
            vel_alignment = short_vel2 * vel_consistency
            
            # Volume Efficiency
            vol_efficiency = data['volume'].iloc[i] / (data['volume'].iloc[i-3] + 1e-8)
            daily_efficiency2 = abs(data['close'].iloc[i-1] - data['close'].iloc[i-2]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1] + 1e-8)
            factor5 = range_quality * vel_alignment * vol_efficiency * daily_efficiency2
            
            # Combine all factors with equal weighting
            combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
            
            factor.loc[current_idx] = combined_factor
            
        except (IndexError, ZeroDivisionError):
            factor.loc[current_idx] = np.nan
    
    return factor
