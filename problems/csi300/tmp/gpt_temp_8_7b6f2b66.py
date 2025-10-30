import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate price changes for fractal efficiency
    data['price_change'] = data['close'].diff()
    data['abs_price_change'] = data['price_change'].abs()
    
    # Calculate volume changes for volume efficiency
    data['volume_change'] = data['volume'].diff()
    data['abs_volume_change'] = data['volume_change'].abs()
    
    # True Range Calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = (data['high'] - data['close'].shift(1)).abs()
    data['tr3'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Initialize factor series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    for i in range(34, len(data)):
        current_data = data.iloc[:i+1]
        
        # 1. Multi-Timeframe Fractal Efficiency Analysis
        fractal_efficiency = {}
        volume_weighted_efficiency = {}
        
        # 3-day calculations
        if i >= 3:
            price_change_sum_3d = current_data['abs_price_change'].iloc[i-2:i+1].sum()
            volume_weighted_sum_3d = (current_data['volume'].iloc[i-2:i+1] * current_data['abs_price_change'].iloc[i-2:i+1]).sum()
            
            if price_change_sum_3d > 0:
                fractal_efficiency['3d'] = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / price_change_sum_3d
            else:
                fractal_efficiency['3d'] = 0
                
            if volume_weighted_sum_3d > 0:
                volume_weighted_efficiency['3d'] = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / volume_weighted_sum_3d
            else:
                volume_weighted_efficiency['3d'] = 0
        
        # 8-day calculations
        if i >= 8:
            price_change_sum_8d = current_data['abs_price_change'].iloc[i-7:i+1].sum()
            volume_weighted_sum_8d = (current_data['volume'].iloc[i-7:i+1] * current_data['abs_price_change'].iloc[i-7:i+1]).sum()
            
            if price_change_sum_8d > 0:
                fractal_efficiency['8d'] = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-8]) / price_change_sum_8d
            else:
                fractal_efficiency['8d'] = 0
                
            if volume_weighted_sum_8d > 0:
                volume_weighted_efficiency['8d'] = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-8]) / volume_weighted_sum_8d
            else:
                volume_weighted_efficiency['8d'] = 0
        
        # 21-day calculations
        if i >= 21:
            price_change_sum_21d = current_data['abs_price_change'].iloc[i-20:i+1].sum()
            volume_weighted_sum_21d = (current_data['volume'].iloc[i-20:i+1] * current_data['abs_price_change'].iloc[i-20:i+1]).sum()
            
            if price_change_sum_21d > 0:
                fractal_efficiency['21d'] = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-21]) / price_change_sum_21d
            else:
                fractal_efficiency['21d'] = 0
                
            if volume_weighted_sum_21d > 0:
                volume_weighted_efficiency['21d'] = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-21]) / volume_weighted_sum_21d
            else:
                volume_weighted_efficiency['21d'] = 0
        
        # 2. Fractal Efficiency Divergence Detection
        divergence = {}
        for timeframe in ['3d', '8d', '21d']:
            if timeframe in fractal_efficiency and timeframe in volume_weighted_efficiency:
                divergence[timeframe] = fractal_efficiency[timeframe] - volume_weighted_efficiency[timeframe]
        
        # Multi-Timeframe Divergence Consistency
        aligned_count = 0
        total_divergences = len(divergence)
        if total_divergences > 0:
            positive_divergences = sum(1 for d in divergence.values() if d > 0)
            negative_divergences = sum(1 for d in divergence.values() if d < 0)
            aligned_count = max(positive_divergences, negative_divergences)
        
        # 3. Range-Based Momentum Enhancement
        momentum = {}
        
        # Short-term momentum (5-day)
        if i >= 5:
            momentum_5d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1) * current_data['true_range'].iloc[i]
            momentum['short'] = momentum_5d
        
        # Medium-term momentum (13-day)
        if i >= 13:
            avg_tr_13d = current_data['true_range'].iloc[i-12:i+1].mean()
            momentum_13d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-13] - 1) * avg_tr_13d
            momentum['medium'] = momentum_13d
        
        # Long-term momentum (34-day)
        if i >= 34:
            avg_tr_34d = current_data['true_range'].iloc[i-33:i+1].mean()
            momentum_34d = (current_data['close'].iloc[i] / current_data['close'].iloc[i-34] - 1) * avg_tr_34d
            momentum['long'] = momentum_34d
        
        # 4. Volume-Intensity Regime Classification
        volume_efficiency = {}
        
        # Volume fractal efficiency calculations
        if i >= 3:
            volume_change_sum_3d = current_data['abs_volume_change'].iloc[i-2:i+1].sum()
            if volume_change_sum_3d > 0:
                volume_efficiency['3d'] = abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-3]) / volume_change_sum_3d
            else:
                volume_efficiency['3d'] = 0
        
        if i >= 8:
            volume_change_sum_8d = current_data['abs_volume_change'].iloc[i-7:i+1].sum()
            if volume_change_sum_8d > 0:
                volume_efficiency['8d'] = abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-8]) / volume_change_sum_8d
            else:
                volume_efficiency['8d'] = 0
        
        if i >= 21:
            volume_change_sum_21d = current_data['abs_volume_change'].iloc[i-20:i+1].sum()
            if volume_change_sum_21d > 0:
                volume_efficiency['21d'] = abs(current_data['volume'].iloc[i] - current_data['volume'].iloc[i-21]) / volume_change_sum_21d
            else:
                volume_efficiency['21d'] = 0
        
        # Determine volume regime
        avg_volume_efficiency = np.mean(list(volume_efficiency.values())) if volume_efficiency else 0
        
        if avg_volume_efficiency > 0.7:
            # High volume intensity regime
            price_weight = 0.4
            volume_weight = 0.6
            momentum_weight_short = 0.6
            momentum_weight_medium = 0.3
            momentum_weight_long = 0.1
        elif avg_volume_efficiency >= 0.3:
            # Medium volume intensity regime
            price_weight = 0.5
            volume_weight = 0.5
            momentum_weight_short = 0.3
            momentum_weight_medium = 0.5
            momentum_weight_long = 0.2
        else:
            # Low volume intensity regime
            price_weight = 0.7
            volume_weight = 0.3
            momentum_weight_short = 0.1
            momentum_weight_medium = 0.3
            momentum_weight_long = 0.6
        
        # 5. Multi-Dimensional Alpha Synthesis
        # Core factor construction
        core_factor = 0
        
        # Combine fractal efficiency divergence with regime weighting
        if divergence:
            avg_divergence = np.mean(list(divergence.values()))
            weighted_divergence = (price_weight * avg_divergence + 
                                 volume_weight * avg_divergence * (1 if avg_divergence > 0 else -1))
            core_factor += weighted_divergence
        
        # Add momentum components with regime-based weighting
        momentum_components = []
        if 'short' in momentum:
            momentum_components.append(momentum_weight_short * momentum['short'])
        if 'medium' in momentum:
            momentum_components.append(momentum_weight_medium * momentum['medium'])
        if 'long' in momentum:
            momentum_components.append(momentum_weight_long * momentum['long'])
        
        if momentum_components:
            momentum_component = np.sum(momentum_components)
            core_factor += momentum_component
        
        # Apply multi-timeframe alignment scoring
        alignment_score = aligned_count / total_divergences if total_divergences > 0 else 0.5
        core_factor *= (1 + 0.5 * alignment_score)  # Amplify aligned signals
        
        # Signal enhancement - suppress contradictory patterns
        if divergence and momentum:
            divergence_sign = np.sign(avg_divergence) if 'avg_divergence' in locals() else 0
            momentum_sign = np.sign(momentum_component) if 'momentum_component' in locals() else 0
            
            if divergence_sign != 0 and momentum_sign != 0 and divergence_sign != momentum_sign:
                # Contradictory patterns - reduce signal strength
                core_factor *= 0.3
        
        factor_values.iloc[i] = core_factor
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    return factor_values
