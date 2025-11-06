import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(20, len(data)):
        current_data = data.iloc[:i+1]
        
        # 1. Measure Liquidity Asymmetry
        # Compute Bidirectional Volume Impact
        volume_sums = []
        up_volume_sums = []
        down_volume_sums = []
        
        for j in range(max(0, i-4), i+1):
            if j > 0:
                volume_sum = current_data['volume'].iloc[max(0, j-4):j+1].sum()
                up_volume = current_data.loc[
                    (current_data.index >= current_data.index[max(0, j-4)]) & 
                    (current_data.index <= current_data.index[j]) & 
                    (current_data['close'] > current_data['close'].shift(1))
                ]['volume'].sum()
                down_volume = current_data.loc[
                    (current_data.index >= current_data.index[max(0, j-4)]) & 
                    (current_data.index <= current_data.index[j]) & 
                    (current_data['close'] < current_data['close'].shift(1))
                ]['volume'].sum()
                
                volume_sums.append(volume_sum)
                up_volume_sums.append(up_volume)
                down_volume_sums.append(down_volume)
        
        if len(volume_sums) > 0 and volume_sums[-1] > 0:
            up_move_absorption = up_volume_sums[-1] / volume_sums[-1]
            down_move_absorption = down_volume_sums[-1] / volume_sums[-1]
            absorption_asymmetry = up_move_absorption - down_move_absorption
        else:
            absorption_asymmetry = 0
        
        # Calculate Price-Volume Elasticity
        if i >= 5:
            price_change_5d = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
            
            if up_volume_sums[-1] > 0:
                elasticity_up = price_change_5d / up_move_absorption
            else:
                elasticity_up = 0
                
            if down_volume_sums[-1] > 0:
                elasticity_down = price_change_5d / down_move_absorption
            else:
                elasticity_down = 0
                
            elasticity_divergence = elasticity_up - elasticity_down
        else:
            elasticity_divergence = 0
        
        # Assess Liquidity Imbalance Quality
        absorption_sign_consistency = 0
        if len(up_volume_sums) >= 5:
            signs = []
            for k in range(len(up_volume_sums)-4, len(up_volume_sums)):
                if k > 0 and k < len(up_volume_sums):
                    vol_sum_k = volume_sums[k]
                    up_vol_k = up_volume_sums[k]
                    down_vol_k = down_volume_sums[k]
                    if vol_sum_k > 0:
                        asym_k = (up_vol_k - down_vol_k) / vol_sum_k
                        asym_prev = (up_volume_sums[k-1] - down_volume_sums[k-1]) / volume_sums[k-1] if k > 0 and volume_sums[k-1] > 0 else 0
                        signs.append(1 if asym_k * asym_prev > 0 else 0)
            absorption_sign_consistency = sum(signs) / len(signs) if signs else 0
        
        # Volume Confirmation (correlation between absolute price change and volume)
        if i >= 4:
            price_changes = []
            volumes = []
            for j in range(i-4, i+1):
                if j > 0:
                    price_changes.append(abs(current_data['close'].iloc[j] - current_data['close'].iloc[j-1]))
                    volumes.append(current_data['volume'].iloc[j])
            if len(price_changes) > 1:
                volume_confirmation = np.corrcoef(price_changes, volumes)[0,1] if not np.isnan(np.corrcoef(price_changes, volumes)[0,1]) else 0
            else:
                volume_confirmation = 0
        else:
            volume_confirmation = 0
        
        liquidity_score = absorption_asymmetry * elasticity_divergence * absorption_sign_consistency
        
        # 2. Detect Regime-Dependent Reversal Patterns
        # Identify Volatility Compression Phases
        if i >= 5:
            current_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            range_5d_ago = current_data['high'].iloc[i-5] - current_data['low'].iloc[i-5]
            range_compression = current_range / range_5d_ago - 1 if range_5d_ago > 0 else 0
            
            current_volume = current_data['volume'].iloc[i]
            volume_5d_ago = current_data['volume'].iloc[i-5]
            volume_compression = current_volume / volume_5d_ago - 1 if volume_5d_ago > 0 else 0
            
            compression_score = range_compression * volume_compression
        else:
            compression_score = 0
        
        # Calculate Mean-Reversion Pressure
        if i >= 4:
            high_4d = current_data['high'].iloc[i-4:i+1].max()
            low_4d = current_data['low'].iloc[i-4:i+1].min()
            mid_point = (high_4d + low_4d) / 2
            price_range = high_4d - low_4d
            
            if price_range > 0:
                extreme_price_deviation = (current_data['close'].iloc[i] - mid_point) / price_range
            else:
                extreme_price_deviation = 0
            
            # Volume Accumulation
            vol_accumulation = down_volume_sums[-1] - up_volume_sums[-1] if len(down_volume_sums) > 0 and len(up_volume_sums) > 0 else 0
            
            reversal_pressure = extreme_price_deviation * vol_accumulation
        else:
            reversal_pressure = 0
        
        # Assess Regime-Adaptive Reversal Strength
        compression_enhanced_reversal = compression_score * reversal_pressure
        
        # Historical Reversal Efficiency (correlation between reversal pressure and next day return)
        if i >= 19:
            reversal_pressures = []
            next_returns = []
            for j in range(i-19, i+1):
                if j >= 4 and j < len(current_data)-1:
                    high_j = current_data['high'].iloc[j-4:j+1].max()
                    low_j = current_data['low'].iloc[j-4:j+1].min()
                    mid_j = (high_j + low_j) / 2
                    range_j = high_j - low_j
                    
                    if range_j > 0:
                        dev_j = (current_data['close'].iloc[j] - mid_j) / range_j
                    else:
                        dev_j = 0
                    
                    # Calculate volume accumulation for period j
                    vol_sum_j = current_data['volume'].iloc[max(0, j-4):j+1].sum()
                    up_vol_j = current_data.loc[
                        (current_data.index >= current_data.index[max(0, j-4)]) & 
                        (current_data.index <= current_data.index[j]) & 
                        (current_data['close'] > current_data['close'].shift(1))
                    ]['volume'].sum()
                    down_vol_j = current_data.loc[
                        (current_data.index >= current_data.index[max(0, j-4)]) & 
                        (current_data.index <= current_data.index[j]) & 
                        (current_data['close'] < current_data['close'].shift(1))
                    ]['volume'].sum()
                    
                    rev_pressure_j = dev_j * (down_vol_j - up_vol_j)
                    next_return = current_data['close'].iloc[j+1] / current_data['close'].iloc[j] - 1
                    
                    reversal_pressures.append(rev_pressure_j)
                    next_returns.append(next_return)
            
            if len(reversal_pressures) > 1 and len(next_returns) > 1:
                historical_reversal_efficiency = np.corrcoef(reversal_pressures, next_returns)[0,1] if not np.isnan(np.corrcoef(reversal_pressures, next_returns)[0,1]) else 0
            else:
                historical_reversal_efficiency = 0
        else:
            historical_reversal_efficiency = 0
        
        regime_reversal_score = compression_enhanced_reversal * historical_reversal_efficiency
        
        # 3. Evaluate Order Flow Imbalance Dynamics
        # Compute Trade Size Polarization
        if i >= 19:
            # Calculate 80th percentile of amount for last 20 days
            amount_percentile_80 = np.percentile(current_data['amount'].iloc[i-19:i+1], 80)
            
            large_buy_amount = 0
            large_sell_amount = 0
            total_amount_5d = 0
            
            for j in range(max(0, i-4), i+1):
                if current_data['amount'].iloc[j] > amount_percentile_80:
                    if j > 0 and current_data['close'].iloc[j] > current_data['close'].iloc[j-1]:
                        large_buy_amount += current_data['amount'].iloc[j]
                    elif j > 0 and current_data['close'].iloc[j] < current_data['close'].iloc[j-1]:
                        large_sell_amount += current_data['amount'].iloc[j]
                
                total_amount_5d += current_data['amount'].iloc[j]
            
            if total_amount_5d > 0:
                large_buy_dominance = large_buy_amount / total_amount_5d
                large_sell_dominance = large_sell_amount / total_amount_5d
                polarization_ratio = large_buy_dominance - large_sell_dominance
            else:
                polarization_ratio = 0
        else:
            polarization_ratio = 0
        
        # Calculate Flow Persistence
        polarization_ratios = []
        for j in range(max(0, i-4), i+1):
            if j >= 19:
                amount_percentile_80_j = np.percentile(current_data['amount'].iloc[j-19:j+1], 80)
                
                large_buy_j = 0
                large_sell_j = 0
                total_amount_j = 0
                
                for k in range(max(0, j-4), j+1):
                    if current_data['amount'].iloc[k] > amount_percentile_80_j:
                        if k > 0 and current_data['close'].iloc[k] > current_data['close'].iloc[k-1]:
                            large_buy_j += current_data['amount'].iloc[k]
                        elif k > 0 and current_data['close'].iloc[k] < current_data['close'].iloc[k-1]:
                            large_sell_j += current_data['amount'].iloc[k]
                    
                    total_amount_j += current_data['amount'].iloc[k]
                
                if total_amount_j > 0:
                    pol_ratio_j = (large_buy_j - large_sell_j) / total_amount_j
                else:
                    pol_ratio_j = 0
                
                polarization_ratios.append(pol_ratio_j)
        
        # Direction Persistence
        direction_persistence = 0
        if len(polarization_ratios) >= 5:
            direction_count = 0
            for k in range(1, len(polarization_ratios)):
                if polarization_ratios[k] * polarization_ratios[k-1] > 0:
                    direction_count += 1
            direction_persistence = direction_count / (len(polarization_ratios) - 1)
        
        # Magnitude Persistence
        if len(polarization_ratios) >= 2:
            magnitude_persistence = np.std(polarization_ratios) / (np.mean(np.abs(polarization_ratios)) + 1e-8)
        else:
            magnitude_persistence = 1
        
        flow_quality = polarization_ratio * direction_persistence / (magnitude_persistence + 1e-8)
        
        # Assess Market Impact Asymmetry
        if i >= 5:
            price_change_magnitude = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-5])
            total_amount_5d = current_data['amount'].iloc[i-4:i+1].sum()
            
            if total_amount_5d > 0:
                impact_efficiency = price_change_magnitude / total_amount_5d
            else:
                impact_efficiency = 0
        else:
            impact_efficiency = 0
        
        asymmetric_impact = impact_efficiency * polarization_ratio
        flow_dynamics_score = flow_quality * asymmetric_impact
        
        # 4. Generate Adaptive Alpha Signal
        # Combine Liquidity and Reversal Signals
        liquidity_driven_reversal = liquidity_score * regime_reversal_score
        flow_enhanced_signal = flow_dynamics_score * compression_score
        asymmetric_alpha = liquidity_driven_reversal * flow_enhanced_signal
        
        # Apply Dynamic Confidence Weighting
        liquidity_confidence = abs(liquidity_score) * absorption_sign_consistency
        reversal_confidence = abs(regime_reversal_score) * abs(historical_reversal_efficiency)
        flow_confidence = abs(flow_dynamics_score) * direction_persistence
        
        # Generate Final Output with Confidence Weighting
        total_confidence = liquidity_confidence + reversal_confidence + flow_confidence + 1e-8
        weighted_alpha = asymmetric_alpha * (liquidity_confidence + reversal_confidence + flow_confidence) / total_confidence
        
        result.iloc[i] = weighted_alpha
    
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
