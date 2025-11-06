import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Volatility Regime Momentum factor
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Define session boundaries (assuming 6.5 hour trading day)
    morning_end_hour = 11.5  # 11:30 AM
    afternoon_start_hour = 13.0  # 1:00 PM
    
    # Helper function to calculate local extrema
    def local_extrema(series, window=20, mode='max'):
        if mode == 'max':
            return series.rolling(window, center=True).max()
        else:
            return series.rolling(window, center=True).min()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 40:  # Need enough history for calculations
            result.iloc[i] = 0
            continue
            
        current_date = data.index[i]
        
        # Get recent data for calculations
        recent_data = data.iloc[max(0, i-39):i+1]
        
        # Morning session calculations (first 2 hours)
        morning_high = recent_data['high'].iloc[-1]  # Simplified - would need intraday data
        morning_low = recent_data['low'].iloc[-1]    # Simplified
        morning_open = recent_data['open'].iloc[-1]  # Simplified
        morning_volume = recent_data['volume'].iloc[-1]  # Simplified
        
        # Afternoon session calculations (last 2 hours)  
        afternoon_high = recent_data['high'].iloc[-1]  # Simplified
        afternoon_low = recent_data['low'].iloc[-1]    # Simplified
        afternoon_close = recent_data['close'].iloc[-1]  # Simplified
        afternoon_volume = recent_data['volume'].iloc[-1]  # Simplified
        
        # Historical morning volatility (20-day median)
        if len(recent_data) >= 20:
            morning_vol_median = np.median([
                abs(recent_data['high'].iloc[j] - recent_data['low'].iloc[j]) 
                for j in range(max(0, len(recent_data)-20), len(recent_data))
            ])
        else:
            morning_vol_median = abs(morning_high - morning_low)
        
        # Historical afternoon volatility (20-day median)
        if len(recent_data) >= 20:
            afternoon_vol_median = np.median([
                abs(recent_data['high'].iloc[j] - recent_data['low'].iloc[j]) 
                for j in range(max(0, len(recent_data)-20), len(recent_data))
            ])
        else:
            afternoon_vol_median = abs(afternoon_high - afternoon_low)
        
        # Volatility regime classification
        morning_vol_level = abs(morning_high - morning_low)
        afternoon_vol_level = abs(afternoon_high - afternoon_low)
        
        morning_high_vol_regime = morning_vol_level > morning_vol_median
        afternoon_high_vol_regime = afternoon_vol_level > afternoon_vol_median
        
        # Morning momentum calculations
        if morning_high != morning_low:
            morning_momentum_strength = (morning_high - morning_open) / (morning_high - morning_low)
        else:
            morning_momentum_strength = 0
            
        # Afternoon momentum calculations
        if afternoon_high != afternoon_low:
            afternoon_momentum_strength = (afternoon_close - afternoon_low) / (afternoon_high - afternoon_low)
        else:
            afternoon_momentum_strength = 0
        
        # Price impact approximations (using volume/efficiency)
        morning_price_impact = morning_volume / max(1, morning_high - morning_low)
        afternoon_price_impact = afternoon_volume / max(1, afternoon_high - afternoon_low)
        
        # Efficiency-adjusted momentum
        morning_efficiency_momentum = morning_momentum_strength / max(0.001, morning_price_impact)
        afternoon_efficiency_momentum = afternoon_momentum_strength / max(0.001, afternoon_price_impact)
        
        # Volume-weighted momentum for low volatility regimes
        if len(recent_data) >= 5:
            morning_vol_median_5d = np.median(recent_data['volume'].iloc[-5:])
            afternoon_vol_median_5d = np.median(recent_data['volume'].iloc[-5:])
        else:
            morning_vol_median_5d = morning_volume
            afternoon_vol_median_5d = afternoon_volume
            
        morning_volume_weighted = morning_momentum_strength * (morning_volume / max(1, morning_vol_median_5d))
        afternoon_volume_weighted = afternoon_momentum_strength * (afternoon_volume / max(1, afternoon_vol_median_5d))
        
        # Breakout detection
        prev_day_high = recent_data['high'].iloc[-2] if len(recent_data) > 1 else morning_high
        morning_breakout_quality = morning_momentum_strength * (morning_volume / max(1, morning_vol_median_5d))
        
        # Support-resistance dynamics
        morning_resistance = local_extrema(recent_data['high'], 20, 'max').iloc[-1]
        morning_support = local_extrema(recent_data['low'], 20, 'min').iloc[-1]
        afternoon_resistance = morning_resistance  # Simplified
        afternoon_support = morning_support  # Simplified
        
        if morning_resistance != morning_support:
            morning_breakout_power = (morning_high - morning_resistance) / (morning_resistance - morning_support)
        else:
            morning_breakout_power = 0
            
        if afternoon_resistance != afternoon_support:
            afternoon_breakout_power = (afternoon_close - afternoon_resistance) / (afternoon_resistance - afternoon_support)
        else:
            afternoon_breakout_power = 0
            
        # Volume-range efficiency
        morning_volume_efficiency = morning_volume / max(1, morning_high - morning_low)
        afternoon_volume_efficiency = afternoon_volume / max(1, afternoon_high - afternoon_low)
        efficiency_regime_shift = afternoon_volume_efficiency - morning_volume_efficiency
        efficiency_momentum_alignment = efficiency_regime_shift * (morning_momentum_strength - afternoon_momentum_strength)
        
        # Multi-regime signal integration
        regime_alignment = 1 if morning_high_vol_regime == afternoon_high_vol_regime else 0
        
        morning_signal_convergence = morning_efficiency_momentum * morning_breakout_quality
        afternoon_signal_convergence = afternoon_efficiency_momentum * afternoon_breakout_power
        intraday_signal_alignment = morning_signal_convergence * afternoon_signal_convergence
        
        # Final factor calculation based on volatility regime
        if morning_high_vol_regime and afternoon_high_vol_regime:
            # High volatility regime
            result.iloc[i] = intraday_signal_alignment * regime_alignment
        elif not morning_high_vol_regime and not afternoon_high_vol_regime:
            # Low volatility regime
            result.iloc[i] = intraday_signal_alignment / max(1, regime_alignment)
        else:
            # Mixed regime
            result.iloc[i] = efficiency_momentum_alignment * np.sign(morning_breakout_power) * np.sign(afternoon_breakout_power)
    
    # Normalize the result
    if len(result) > 0:
        result = (result - result.mean()) / max(0.001, result.std())
    
    return result
