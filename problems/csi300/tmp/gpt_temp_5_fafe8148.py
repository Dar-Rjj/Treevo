import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Convergence with Regime-Adaptive Volume Confirmation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(8, len(df)):
        current_data = df.iloc[:i+1]
        
        # Momentum Convergence Framework
        # Short-Term Momentum (3-day)
        if i >= 3:
            short_return = current_data['close'].iloc[i] / current_data['close'].iloc[i-3] - 1
            short_volatility = current_data['close'].iloc[i-4:i+1].pct_change().std()
            short_momentum = short_return / (short_volatility + 1e-8) if short_volatility != 0 else 0
        else:
            short_momentum = 0
        
        # Medium-Term Momentum (5-day)
        if i >= 5:
            medium_return = current_data['close'].iloc[i] / current_data['close'].iloc[i-5] - 1
            medium_volatility = current_data['close'].iloc[i-7:i+1].pct_change().std()
            medium_momentum = medium_return / (medium_volatility + 1e-8) if medium_volatility != 0 else 0
        else:
            medium_momentum = 0
        
        # Convergence Analysis
        direction_alignment = np.sign(short_momentum) == np.sign(medium_momentum)
        convergence_strength = np.sign(short_momentum) * np.sign(medium_momentum)
        magnitude_reinforcement = (abs(short_momentum) + abs(medium_momentum)) / 2
        
        # Volume Confirmation System
        # Volume Momentum Analysis
        if i >= 3:
            vol_3d_momentum = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-3] - 1
        else:
            vol_3d_momentum = 0
            
        if i >= 5:
            vol_5d_momentum = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5] - 1
        else:
            vol_5d_momentum = 0
            
        volume_trend_consistency = np.sign(vol_3d_momentum) == np.sign(vol_5d_momentum)
        
        # Price-Volume Divergence
        price_vol_div_3d = np.sign(short_momentum) != np.sign(vol_3d_momentum) if i >= 3 else False
        price_vol_div_5d = np.sign(medium_momentum) != np.sign(vol_5d_momentum) if i >= 5 else False
        overall_divergence = price_vol_div_3d or price_vol_div_5d
        
        # Volume Persistence
        consecutive_high_volume = 0
        if i >= 2:
            for j in range(min(5, i+1)):
                lookback_idx = i - j
                if lookback_idx >= 2:
                    vol_window = current_data['volume'].iloc[lookback_idx-2:lookback_idx+1]
                    if current_data['volume'].iloc[lookback_idx] > vol_window.mean():
                        consecutive_high_volume += 1
                    else:
                        break
        
        if i >= 2:
            volume_acceleration = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-2:i+1].mean()
        else:
            volume_acceleration = 1
            
        if i >= 4:
            vol_window = current_data['volume'].iloc[i-4:i+1]
            volume_stability = vol_window.mean() / (vol_window.std() + 1e-8)
        else:
            volume_stability = 1
        
        # Market Regime Classification
        # Volatility Regime
        daily_range = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / current_data['close'].iloc[i]
        
        if i >= 4:
            range_5d_avg = sum((current_data['high'].iloc[j] - current_data['low'].iloc[j]) / current_data['close'].iloc[j] 
                              for j in range(i-4, i+1)) / 5
        else:
            range_5d_avg = daily_range
            
        if daily_range > 1.5 * range_5d_avg:
            volatility_regime = 'high'
            momentum_multiplier = 0.7
            volume_weight = 0.6
        elif daily_range < 0.6 * range_5d_avg:
            volatility_regime = 'low'
            momentum_multiplier = 1.3
            volume_weight = 1.5
        else:
            volatility_regime = 'normal'
            momentum_multiplier = 1.0
            volume_weight = 1.0
        
        # Volume Regime
        if i >= 4:
            volume_surge = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-4:i+1].mean()
        else:
            volume_surge = 1
            
        if volume_surge > 1.8:
            volume_regime = 'high'
            volume_weight_regime = 1.4
            persistence_requirement = 0.8
        elif volume_surge < 0.5:
            volume_regime = 'low'
            volume_weight_regime = 0.6
            persistence_requirement = 1.2
        else:
            volume_regime = 'normal'
            volume_weight_regime = 1.0
            persistence_requirement = 1.0
        
        # Combined Regime Effects
        volatility_weight_product = momentum_multiplier * volume_weight
        regime_multiplier = volatility_weight_product * volume_weight_regime
        
        # Alpha Factor Construction
        # Base Momentum Signal
        weighted_momentum = (short_momentum * 0.4 + medium_momentum * 0.6)
        
        if direction_alignment:
            convergence_multiplier = 1.5
        else:
            convergence_multiplier = 0.8
            
        base_momentum = weighted_momentum * convergence_multiplier * (1 + magnitude_reinforcement)
        
        # Volume Confirmation Score
        volume_confirmation = 1.0
        
        # Volume momentum alignment
        if volume_trend_consistency:
            volume_confirmation *= 1.2
        else:
            volume_confirmation *= 0.9
            
        # Price-volume divergence penalty
        if overall_divergence:
            volume_confirmation *= 0.7
            
        # Volume persistence bonus
        volume_confirmation *= (1.0 + (consecutive_high_volume * 0.1))
        
        # Volume stability adjustment
        volume_confirmation *= (volume_stability * 0.5)
        
        # Final Signal Integration
        volume_adjusted_momentum = base_momentum * volume_confirmation
        regime_adjusted_signal = volume_adjusted_momentum * regime_multiplier
        
        # Persistence filter
        if consecutive_high_volume >= persistence_requirement:
            alpha.iloc[i] = regime_adjusted_signal
        else:
            alpha.iloc[i] = 0
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
