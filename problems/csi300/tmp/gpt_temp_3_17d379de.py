import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate novel alpha factor combining fractal dimension, range efficiency,
    volume clustering, and temporal patterns for stock return prediction.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need sufficient history for calculations
            factor_values.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Multi-scale Fractal Dimension Calculation
        fractal_scores = []
        for window in [2, 5, 8]:
            if i >= window:
                window_data = current_data.iloc[i-window+1:i+1]
                high_low_range = (window_data['high'] - window_data['low']).abs()
                close_range = window_data['close'].diff().abs()
                
                # Calculate roughness measure
                if high_low_range.sum() > 0:
                    roughness = (close_range.sum() / high_low_range.sum()) 
                    fractal_dim = 2 - np.log(1 + roughness) / np.log(window)
                    fractal_scores.append(fractal_dim)
                else:
                    fractal_scores.append(1.0)
        
        fractal_score = np.mean(fractal_scores) if fractal_scores else 1.0
        
        # 2. Range Efficiency Asymmetry
        current_row = current_data.iloc[-1]
        high_low_range = current_row['high'] - current_row['low']
        
        if high_low_range > 0:
            upside_efficiency = (current_row['high'] - current_row['open']) / high_low_range
            downside_efficiency = (current_row['open'] - current_row['low']) / high_low_range
            efficiency_divergence = upside_efficiency - downside_efficiency
        else:
            efficiency_divergence = 0
        
        # 3. Volume Clustering Dynamics
        recent_volume = current_data['volume'].iloc[max(0, i-4):i+1]
        price_extremes_volume = 0
        
        for j in range(max(0, i-4), i+1):
            row = current_data.iloc[j]
            high_low_range_j = row['high'] - row['low']
            if high_low_range_j > 0:
                # Volume near price extremes
                high_proximity = (row['high'] - row['close']) / high_low_range_j
                low_proximity = (row['close'] - row['low']) / high_low_range_j
                if high_proximity < 0.1 or low_proximity < 0.1:
                    price_extremes_volume += row['volume']
        
        total_recent_volume = recent_volume.sum()
        volume_clustering = price_extremes_volume / total_recent_volume if total_recent_volume > 0 else 0
        
        # 4. Bid-Ask Imbalance Proxy
        if high_low_range > 0:
            buying_pressure = ((current_row['close'] - current_row['low']) / high_low_range) * current_row['volume']
            selling_pressure = ((current_row['high'] - current_row['close']) / high_low_range) * current_row['volume']
            
            if selling_pressure > 0:
                pressure_imbalance = (buying_pressure - selling_pressure) / (buying_pressure + selling_pressure)
            else:
                pressure_imbalance = 0
        else:
            pressure_imbalance = 0
        
        # 5. Temporal Pattern - Morning vs Afternoon (using opening gap as proxy)
        if i > 0:
            prev_close = current_data.iloc[i-1]['close']
            gap_strength = abs(current_row['open'] - prev_close) / high_low_range if high_low_range > 0 else 0
        else:
            gap_strength = 0
        
        # 6. Price Elasticity - Response to volume shocks
        if i >= 5:
            recent_volume_change = current_data['volume'].iloc[i] / (current_data['volume'].iloc[i-5:i].mean() + 1e-8)
            price_change = current_data['close'].iloc[i] / (current_data['close'].iloc[i-1] + 1e-8) - 1
            volume_shock_response = price_change / (recent_volume_change + 1e-8)
        else:
            volume_shock_response = 0
        
        # 7. Cross-Dimensional Pattern Integration
        # Fractal Efficiency Momentum
        fractal_efficiency = fractal_score * (1 + efficiency_divergence)
        
        # Elasticity-Regime Dynamics
        elasticity_regime = volume_shock_response * pressure_imbalance
        
        # Temporal-Fractal Convergence
        temporal_fractal = gap_strength * fractal_score
        
        # 8. Dynamic Signal Processing with Pattern-Adaptive Weighting
        # Weight components based on their recent persistence
        weights = []
        components = []
        
        # Fractal persistence (using recent fractal stability)
        if i >= 10:
            recent_fractals = []
            for k in range(max(0, i-9), i+1):
                if k >= 2:
                    window_data_k = current_data.iloc[k-1:k+1]
                    range_k = (window_data_k['high'] - window_data_k['low']).abs()
                    close_range_k = window_data_k['close'].diff().abs()
                    if range_k.sum() > 0:
                        roughness_k = close_range_k.sum() / range_k.sum()
                        fractal_k = 2 - np.log(1 + roughness_k) / np.log(2)
                        recent_fractals.append(fractal_k)
            
            if recent_fractals:
                fractal_persistence = 1 - np.std(recent_fractals) / (np.mean(recent_fractals) + 1e-8)
                weights.append(max(0, fractal_persistence))
                components.append(fractal_efficiency)
        
        # Volume clustering persistence
        if i >= 8:
            recent_clustering = []
            for k in range(max(0, i-7), i+1):
                vol_data = current_data['volume'].iloc[max(0, k-4):k+1]
                extremes_vol = 0
                for m in range(max(0, k-4), k+1):
                    row_m = current_data.iloc[m]
                    range_m = row_m['high'] - row_m['low']
                    if range_m > 0:
                        high_prox = (row_m['high'] - row_m['close']) / range_m
                        low_prox = (row_m['close'] - row_m['low']) / range_m
                        if high_prox < 0.1 or low_prox < 0.1:
                            extremes_vol += row_m['volume']
                cluster_val = extremes_vol / vol_data.sum() if vol_data.sum() > 0 else 0
                recent_clustering.append(cluster_val)
            
            if recent_clustering:
                cluster_persistence = 1 - np.std(recent_clustering) / (np.mean(recent_clustering) + 1e-8)
                weights.append(max(0, cluster_persistence))
                components.append(volume_clustering * pressure_imbalance)
        
        # Efficiency persistence
        if i >= 6:
            recent_efficiency = []
            for k in range(max(0, i-5), i+1):
                row_k = current_data.iloc[k]
                range_k = row_k['high'] - row_k['low']
                if range_k > 0:
                    up_eff = (row_k['high'] - row_k['open']) / range_k
                    down_eff = (row_k['open'] - row_k['low']) / range_k
                    eff_div = up_eff - down_eff
                    recent_efficiency.append(eff_div)
            
            if recent_efficiency:
                efficiency_persistence = 1 - np.std(recent_efficiency) / (np.mean(recent_efficiency) + 1e-8)
                weights.append(max(0, efficiency_persistence))
                components.append(elasticity_regime)
        
        # Temporal pattern weight (using gap strength consistency)
        if i >= 5:
            recent_gaps = []
            for k in range(max(0, i-4), i+1):
                if k > 0:
                    prev_close_k = current_data.iloc[k-1]['close']
                    range_k = current_data.iloc[k]['high'] - current_data.iloc[k]['low']
                    if range_k > 0:
                        gap_k = abs(current_data.iloc[k]['open'] - prev_close_k) / range_k
                        recent_gaps.append(gap_k)
            
            if recent_gaps:
                temporal_persistence = 1 - np.std(recent_gaps) / (np.mean(recent_gaps) + 1e-8)
                weights.append(max(0, temporal_persistence))
                components.append(temporal_fractal)
        
        # Calculate final factor value with adaptive weighting
        if weights and components:
            normalized_weights = np.array(weights) / (sum(weights) + 1e-8)
            factor_value = np.dot(normalized_weights, components)
        else:
            # Fallback: simple combination when insufficient history
            factor_value = (fractal_efficiency + volume_clustering * pressure_imbalance + 
                          elasticity_regime + temporal_fractal) / 4
        
        factor_values.iloc[i] = factor_value
    
    return factor_values
