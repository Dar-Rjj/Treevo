import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required components
    for i in range(8, len(df)):
        # Multi-Scale Fractal Dynamics
        # Price Fractal Components
        if i >= 3:
            nonlinear_momentum = ((df.iloc[i]['close'] - df.iloc[i-2]['close'])**2 / (df.iloc[i-2]['close'] + 0.001) - 
                                (df.iloc[i-1]['close'] - df.iloc[i-3]['close'])**2 / (df.iloc[i-3]['close'] + 0.001))
        else:
            nonlinear_momentum = 0
            
        # Intraday Fractal Dimension
        if i >= 2:
            high_range_current = df.iloc[i]['high'] - df.iloc[i]['low']
            high_range_past = np.mean([df.iloc[j]['high'] - df.iloc[j]['low'] for j in range(i-2, i+1)])
            if high_range_current > 0 and high_range_past > 0:
                intraday_fractal_dim = np.log(high_range_current) / np.log(high_range_past)
            else:
                intraday_fractal_dim = 0.5
        else:
            intraday_fractal_dim = 0.5
            
        # Fractal Persistence
        if i >= 6:
            persistence_count = 0
            for j in range(i-4, i+1):
                if j >= 2:
                    sign_current = np.sign(df.iloc[j]['close'] - df.iloc[j-1]['close'])
                    sign_prev = np.sign(df.iloc[j-1]['close'] - df.iloc[j-2]['close'])
                    if sign_current == sign_prev:
                        persistence_count += 1
            fractal_persistence = persistence_count
        else:
            fractal_persistence = 0
            
        # Volume Fractal Components
        if i >= 3:
            volume_power = (df.iloc[i]['volume']**0.7 / (df.iloc[i-3]['volume']**0.7 + 1e-8)) - 1
        else:
            volume_power = 0
            
        # Volume Fractal Dimension
        if i >= 3:
            vol_current = df.iloc[i]['volume']
            vol_avg = np.mean([df.iloc[j]['volume'] for j in range(i-3, i+1)])
            if vol_current > 0 and vol_avg > 0:
                volume_fractal_dim = np.log(vol_current) / np.log(vol_avg)
            else:
                volume_fractal_dim = 0.5
        else:
            volume_fractal_dim = 0.5
            
        # Volume Fractal Momentum
        if i >= 3:
            vol_frac_momentum = (df.iloc[i]['volume'] / (df.iloc[i-2]['volume'] + 1e-8) - 
                               df.iloc[i-1]['volume'] / (df.iloc[i-3]['volume'] + 1e-8))
        else:
            vol_frac_momentum = 0
            
        # Cross-Fractal Interactions
        price_volume_fractal = nonlinear_momentum * volume_power
        fractal_dim_alignment = intraday_fractal_dim * volume_fractal_dim
        vol_price_fractal_corr = np.sign(vol_frac_momentum) * np.sign(nonlinear_momentum)
        
        # Asymmetric Pressure Dynamics
        # Pressure Components
        high_low_range = df.iloc[i]['high'] - df.iloc[i]['low']
        if high_low_range > 0:
            buy_pressure = ((df.iloc[i]['close'] - df.iloc[i]['low'])**1.2 / high_low_range**1.2)
            sell_pressure = ((df.iloc[i]['high'] - df.iloc[i]['close'])**1.2 / high_low_range**1.2)
        else:
            buy_pressure = 0.5
            sell_pressure = 0.5
        pressure_asymmetry = buy_pressure - sell_pressure
        
        # Volume-Weighted Efficiency
        if high_low_range > 0:
            vol_weighted_return = (df.iloc[i]['close'] - df.iloc[i]['open']) * df.iloc[i]['volume'] / high_low_range
        else:
            vol_weighted_return = 0
            
        if i >= 4:
            vol_avg_5 = np.mean([df.iloc[j]['volume'] for j in range(i-4, i+1)])
            if vol_avg_5 > 0:
                vol_efficiency_ratio = (df.iloc[i]['close'] - df.iloc[i]['open']) / (df.iloc[i]['volume'] / vol_avg_5)
            else:
                vol_efficiency_ratio = 0
        else:
            vol_efficiency_ratio = 0
            
        vol_price_consistency = np.sign(df.iloc[i]['close'] - df.iloc[i]['open']) * np.sign(df.iloc[i]['volume'] - df.iloc[i-1]['volume'])
        
        # Multi-day Pressure Patterns
        if i >= 1:
            if (df.iloc[i]['high'] - df.iloc[i]['low']) > 0:
                buy_pressure_t1 = ((df.iloc[i-1]['close'] - df.iloc[i-1]['low'])**1.2 / 
                                 (df.iloc[i-1]['high'] - df.iloc[i-1]['low'])**1.2)
            else:
                buy_pressure_t1 = 0.5
                
            if (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) > 0:
                sell_pressure_t1 = ((df.iloc[i-1]['high'] - df.iloc[i-1]['close'])**1.2 / 
                                  (df.iloc[i-1]['high'] - df.iloc[i-1]['low'])**1.2)
            else:
                sell_pressure_t1 = 0.5
                
            two_day_buy = buy_pressure + buy_pressure_t1
            two_day_sell = sell_pressure + sell_pressure_t1
            pressure_momentum = two_day_buy - two_day_sell
        else:
            pressure_momentum = 0
            
        # Regime Detection & Switching
        # Fractal Regime Detection
        high_fractality_regime = (intraday_fractal_dim > 0.8) and (volume_fractal_dim > 0.7)
        low_fractality_regime = (intraday_fractal_dim < 0.4) and (volume_fractal_dim < 0.3)
        fractal_transition = abs(intraday_fractal_dim - volume_fractal_dim) > 0.3
        
        # Momentum Regime Detection
        momentum_regime = np.sign(nonlinear_momentum) * np.sign(price_volume_fractal)
        
        if i >= 5:
            volatility_ratio = ((df.iloc[i]['high'] - df.iloc[i]['low'])**1.5 / 
                              (df.iloc[i-5]['high'] - df.iloc[i-5]['low'])**1.5) - 1
            volatility_regime = 1 if volatility_ratio > 0.1 else -1
        else:
            volatility_regime = 0
            
        volume_regime = 1 if volume_power > 0.05 else (-1 if volume_power < -0.05 else 0)
        
        # Regime Transition Signals
        momentum_transition = abs(nonlinear_momentum) * (1 + abs(price_volume_fractal))
        fractal_transition_signal = fractal_transition * vol_price_fractal_corr
        volume_transition = volume_power * vol_frac_momentum
        
        # Regime-Adaptive Signal Construction
        # High Fractality Signals
        fractal_momentum_signal = nonlinear_momentum * vol_frac_momentum
        volume_fractal_efficiency = vol_efficiency_ratio * intraday_fractal_dim
        fractal_pressure_signal = buy_pressure * volume_fractal_dim - sell_pressure * volume_power
        
        # Low Fractality Signals
        vol_price_efficiency_signal = vol_weighted_return * vol_efficiency_ratio
        pressure_persistence_signal = pressure_momentum * fractal_persistence
        volume_consistency_signal = vol_price_consistency * vol_frac_momentum
        
        # Regime-Switching Integration
        high_fractality_alpha = (fractal_momentum_signal * volume_fractal_efficiency * 
                               fractal_pressure_signal)
        low_fractality_alpha = (vol_price_efficiency_signal * pressure_persistence_signal * 
                              volume_consistency_signal)
        
        adaptive_alpha = high_fractality_alpha if high_fractality_regime else low_fractality_alpha
        
        # Multi-Timeframe Enhancement
        # Short-term Dynamics (1-3 days)
        if i >= 3:
            vol_frac_acceleration = vol_frac_momentum - (df.iloc[i-1]['volume'] / (df.iloc[i-3]['volume'] + 1e-8))
            if i >= 2:
                price_frac_acceleration = (nonlinear_momentum - 
                                         (df.iloc[i-1]['close'] - df.iloc[i-2]['close'])**2 / 
                                         (df.iloc[i-2]['close'] + 0.001))
            else:
                price_frac_acceleration = 0
            short_term_alignment = np.sign(vol_frac_acceleration) * np.sign(price_frac_acceleration)
        else:
            short_term_alignment = 0
            
        # Medium-term Dynamics (4-8 days)
        if i >= 8:
            vol_frac_persistence = 0
            for j in range(i-4, i+1):
                if j >= 3:
                    vol_current_j = df.iloc[j]['volume']
                    vol_avg_j = np.mean([df.iloc[k]['volume'] for k in range(j-3, j+1)])
                    if vol_current_j > 0 and vol_avg_j > 0:
                        vol_dim_j = np.log(vol_current_j) / np.log(vol_avg_j)
                        if vol_dim_j > 0.5:
                            vol_frac_persistence += 1
            
            intraday_fractal_dims = []
            for j in range(i-4, i+1):
                if j >= 2:
                    high_range_current_j = df.iloc[j]['high'] - df.iloc[j]['low']
                    high_range_past_j = np.mean([df.iloc[k]['high'] - df.iloc[k]['low'] for k in range(j-2, j+1)])
                    if high_range_current_j > 0 and high_range_past_j > 0:
                        intraday_dim_j = np.log(high_range_current_j) / np.log(high_range_past_j)
                        intraday_fractal_dims.append(intraday_dim_j)
            
            if len(intraday_fractal_dims) > 1:
                price_fractal_stability = np.std(intraday_fractal_dims)
            else:
                price_fractal_stability = 0
                
            medium_term_consistency = vol_frac_persistence * (1 - price_fractal_stability)
        else:
            medium_term_consistency = 0
            
        # Final Alpha Construction
        short_term_enhanced_alpha = adaptive_alpha * short_term_alignment
        medium_term_enhanced_alpha = adaptive_alpha * medium_term_consistency
        
        final_alpha = (short_term_enhanced_alpha * medium_term_enhanced_alpha * 
                     fractal_dim_alignment)
        
        result.iloc[i] = final_alpha
        
    # Fill NaN values with 0
    result = result.fillna(0)
    
    return result
