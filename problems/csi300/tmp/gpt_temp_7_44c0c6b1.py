import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Efficiency Component
    # Price-Range Momentum Efficiency
    net_move = data['close'] - data['close'].shift(5)
    total_move = (abs(data['close'] - data['close'].shift(1)) + 
                  abs(data['close'].shift(1) - data['close'].shift(2)) + 
                  abs(data['close'].shift(2) - data['close'].shift(3)) + 
                  abs(data['close'].shift(3) - data['close'].shift(4)) + 
                  abs(data['close'].shift(4) - data['close'].shift(5)))
    efficiency_ratio = net_move / (total_move + 1e-8)
    
    # Volatility-Adjusted Momentum
    short_term_momentum = data['close'] / data['close'].shift(10) - 1
    
    # Recent Volatility (5-day)
    recent_range = ((data['high'] - data['low']) + 
                   (data['high'].shift(1) - data['low'].shift(1)) + 
                   (data['high'].shift(2) - data['low'].shift(2)) + 
                   (data['high'].shift(3) - data['low'].shift(3)) + 
                   (data['high'].shift(4) - data['low'].shift(4))) / 5
    
    # Historical Volatility (20-day)
    historical_range = ((data['high'] - data['low']) + 
                       (data['high'].shift(1) - data['low'].shift(1)) + 
                       (data['high'].shift(2) - data['low'].shift(2)) + 
                       (data['high'].shift(3) - data['low'].shift(3)) + 
                       (data['high'].shift(4) - data['low'].shift(4)) + 
                       (data['high'].shift(5) - data['low'].shift(5)) + 
                       (data['high'].shift(6) - data['low'].shift(6)) + 
                       (data['high'].shift(7) - data['low'].shift(7)) + 
                       (data['high'].shift(8) - data['low'].shift(8)) + 
                       (data['high'].shift(9) - data['low'].shift(9)) + 
                       (data['high'].shift(10) - data['low'].shift(10)) + 
                       (data['high'].shift(11) - data['low'].shift(11)) + 
                       (data['high'].shift(12) - data['low'].shift(12)) + 
                       (data['high'].shift(13) - data['low'].shift(13)) + 
                       (data['high'].shift(14) - data['low'].shift(14)) + 
                       (data['high'].shift(15) - data['low'].shift(15)) + 
                       (data['high'].shift(16) - data['low'].shift(16)) + 
                       (data['high'].shift(17) - data['low'].shift(17)) + 
                       (data['high'].shift(18) - data['low'].shift(18)) + 
                       (data['high'].shift(19) - data['low'].shift(19))) / 20
    
    volatility_ratio = recent_range / (historical_range + 1e-8)
    volatility_adjusted_momentum = short_term_momentum / (volatility_ratio + 1e-8)
    
    # Volume Divergence Component
    # Volume-Price Acceleration
    current_velocity = data['close'] / data['close'].shift(1) - 1
    previous_velocity = data['close'].shift(1) / data['close'].shift(2) - 1
    acceleration = current_velocity - previous_velocity
    volume_acceleration = acceleration * data['volume']
    
    # Volume Cluster Proximity
    def calculate_cluster_proximity(data, window=20):
        cluster_proximity = pd.Series(index=data.index, dtype=float)
        
        for i in range(window, len(data)):
            # Get volume window
            volume_window = data['volume'].iloc[i-window:i+1]
            # Calculate 80th percentile threshold
            volume_threshold = volume_window.quantile(0.8)
            # Get high-volume price levels
            high_volume_mask = volume_window >= volume_threshold
            high_volume_prices = data['close'].iloc[i-window:i+1][high_volume_mask]
            
            if len(high_volume_prices) > 0:
                current_price = data['close'].iloc[i]
                # Find nearest high-volume price level
                distances = abs(high_volume_prices - current_price)
                min_distance = distances.min()
                # Compute inverse distance with direction
                nearest_price = high_volume_prices[distances.idxmin()]
                direction = 1 if nearest_price > current_price else -1
                cluster_proximity.iloc[i] = direction / (min_distance + 1e-8)
            else:
                cluster_proximity.iloc[i] = 0
        
        return cluster_proximity
    
    cluster_proximity = calculate_cluster_proximity(data)
    
    # Signal Integration
    # Efficiency-Volume Interaction
    efficiency_volume_interaction = efficiency_ratio * volume_acceleration / (volatility_ratio + 1e-8)
    
    # Cluster Proximity Weighting
    cluster_weighted_signal = efficiency_volume_interaction * cluster_proximity
    
    # Multi-Timeframe Confirmation
    # Short-Term Consistency Check
    three_day_momentum = data['close'] / data['close'].shift(3) - 1
    five_day_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Directional consistency factor
    consistency_factor = np.sign(three_day_momentum * five_day_momentum) * np.sqrt(abs(three_day_momentum * five_day_momentum))
    
    # Final Signal Generation
    final_signal = cluster_weighted_signal * consistency_factor
    
    # Apply mean-reversion weighting for extreme values
    signal_abs = abs(final_signal)
    signal_mean = signal_abs.rolling(window=20, min_periods=10).mean()
    signal_std = signal_abs.rolling(window=20, min_periods=10).std()
    
    # Z-score based mean reversion weighting
    z_score = (signal_abs - signal_mean) / (signal_std + 1e-8)
    mean_reversion_weight = 1 / (1 + abs(z_score))
    
    final_factor = final_signal * mean_reversion_weight
    
    return final_factor
