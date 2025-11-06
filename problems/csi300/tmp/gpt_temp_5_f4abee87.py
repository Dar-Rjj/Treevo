import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Composite Fractal Microstructure Factor combining multiple alpha insights
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Base parameters
    short_window = 5
    medium_window = 10
    long_window = 15
    
    for i in range(max(long_window, 15), len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Fractal Momentum Divergence
        # Short-term turnover efficiency
        if i >= short_window:
            short_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-short_window]) / \
                           (current_data['high'].iloc[i-short_window:i+1].max() - current_data['low'].iloc[i-short_window:i+1].min())
        else:
            short_momentum = 0
            
        # Medium-term turnover efficiency
        if i >= medium_window:
            medium_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-medium_window]) / \
                            (current_data['high'].iloc[i-medium_window:i+1].max() - current_data['low'].iloc[i-medium_window:i+1].min())
        else:
            medium_momentum = 0
            
        # Long-term turnover efficiency
        if i >= long_window:
            long_momentum = (current_data['close'].iloc[i] - current_data['close'].iloc[i-long_window]) / \
                          (current_data['high'].iloc[i-long_window:i+1].max() - current_data['low'].iloc[i-long_window:i+1].min())
        else:
            long_momentum = 0
            
        # Fractal momentum divergence
        fractal_divergence = (short_momentum - medium_momentum) + (medium_momentum - long_momentum)
        
        # 2. Microstructure Entropy Dynamics
        # Order flow imbalance entropy approximation
        if i >= 1:
            price_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if price_range > 0:
                buying_pressure = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / price_range
                selling_pressure = (current_data['high'].iloc[i] - current_data['close'].iloc[i]) / price_range
                
                # Entropy calculation
                p1 = buying_pressure / (buying_pressure + selling_pressure) if (buying_pressure + selling_pressure) > 0 else 0.5
                p2 = 1 - p1
                
                if p1 > 0 and p2 > 0:
                    entropy = - (p1 * np.log(p1) + p2 * np.log(p2))
                else:
                    entropy = 0
            else:
                entropy = 0
        else:
            entropy = 0
            
        # Trade size complexity (using volume/amount ratio as proxy)
        if current_data['amount'].iloc[i] > 0:
            trade_size_complexity = current_data['volume'].iloc[i] / current_data['amount'].iloc[i]
        else:
            trade_size_complexity = 0
            
        microstructure_entropy = entropy * (1 + trade_size_complexity)
        
        # 3. Gap-Regime Elasticity
        if i >= 1:
            raw_gap = (current_data['open'].iloc[i] / current_data['close'].iloc[i-1]) - 1
            
            # Gap sustainability
            gap_magnitude = abs(current_data['open'].iloc[i] - current_data['close'].iloc[i-1])
            if gap_magnitude > 0:
                gap_sustainability = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / gap_magnitude
            else:
                gap_sustainability = 0
                
            # Gap fill efficiency
            daily_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if daily_range > 0:
                gap_fill_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / daily_range
            else:
                gap_fill_efficiency = 0
                
            # Volume elasticity
            price_change_pct = abs(current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / current_data['close'].iloc[i-1] if current_data['close'].iloc[i-1] > 0 else 0
            range_pct = daily_range / current_data['close'].iloc[i] if current_data['close'].iloc[i] > 0 else 0
            if range_pct > 0:
                daily_elasticity = price_change_pct / range_pct
            else:
                daily_elasticity = 0
                
            gap_elasticity_score = (abs(raw_gap) + abs(gap_sustainability) + gap_fill_efficiency + daily_elasticity) / 4
        else:
            gap_elasticity_score = 0
            
        # 4. Volatility-Impact Asymmetry
        if i >= 1:
            # Micro-impact
            daily_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if daily_range > 0:
                micro_impact = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / daily_range
                short_impact = (current_data['close'].iloc[i] - current_data['close'].iloc[i-1]) / daily_range
            else:
                micro_impact = 0
                short_impact = 0
                
            # Medium-impact (3-day)
            if i >= 3:
                medium_range = current_data['high'].iloc[i-2:i+1].max() - current_data['low'].iloc[i-2:i+1].min()
                if medium_range > 0:
                    medium_impact = (current_data['close'].iloc[i] - current_data['close'].iloc[i-3]) / medium_range
                else:
                    medium_impact = 0
            else:
                medium_impact = 0
                
            impact_asymmetry = (micro_impact + short_impact + medium_impact) / 3
        else:
            impact_asymmetry = 0
            
        # 5. Volume-Entropy Pressure
        if i >= 1:
            daily_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if daily_range > 0:
                buying_pressure_vol = ((current_data['close'].iloc[i] - current_data['low'].iloc[i]) / daily_range) * current_data['volume'].iloc[i]
                selling_pressure_vol = ((current_data['high'].iloc[i] - current_data['close'].iloc[i]) / daily_range) * current_data['volume'].iloc[i]
                net_pressure = buying_pressure_vol - selling_pressure_vol
            else:
                net_pressure = 0
                
            # Volume concentration (5-day)
            if i >= 5:
                volume_sum = sum(current_data['volume'].iloc[i-5:i])
                if volume_sum > 0:
                    volume_concentration = current_data['volume'].iloc[i] / volume_sum
                else:
                    volume_concentration = 0
            else:
                volume_concentration = 0
                
            volume_pressure = net_pressure * (1 + volume_concentration)
        else:
            volume_pressure = 0
            
        # Composite Factor Calculation
        # Apply microstructure entropy weighting to fractal divergence
        entropy_weighted_momentum = fractal_divergence * (1 - microstructure_entropy)
        
        # Incorporate gap-regime elasticity adjustment
        elasticity_adjusted = entropy_weighted_momentum * (1 + gap_elasticity_score)
        
        # Integrate volatility-impact asymmetry
        impact_integrated = elasticity_adjusted * (1 + impact_asymmetry)
        
        # Validate with volume-entropy pressure
        if volume_pressure != 0:
            pressure_validated = impact_integrated * np.sign(volume_pressure) * min(1, abs(volume_pressure) / 1e6)
        else:
            pressure_validated = impact_integrated
            
        result.iloc[i] = pressure_validated
        
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
