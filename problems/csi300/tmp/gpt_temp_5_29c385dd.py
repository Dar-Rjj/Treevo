import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Fractal Compression Breakout Asymmetry Factor
    Combines fractal range compression patterns with breakout asymmetry and volume dynamics
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Multi-timeframe periods
    periods = [5, 8, 13]
    
    for i in range(max(periods), len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Fractal Compression Period Analysis
        fractal_ranges = {}
        compression_signals = {}
        
        for period in periods:
            # Calculate fractal range: (High-Low)/Close
            recent_data = current_data.iloc[-period:]
            current_range = (recent_data['high'].max() - recent_data['low'].min()) / current_data['close'].iloc[-1]
            fractal_ranges[period] = current_range
            
            # Detect range compression: current range < 5-day range average
            if period == 5:
                range_avg = np.mean([(current_data['high'].iloc[j] - current_data['low'].iloc[j]) / current_data['close'].iloc[j] 
                                   for j in range(i-4, i+1)])
                compression_signals[period] = current_range < range_avg
            else:
                compression_signals[period] = current_range < fractal_ranges[5]
        
        # Multi-timeframe compression alignment
        compression_alignment = sum(compression_signals.values()) / len(periods)
        
        # 2. Fractal Breakout Asymmetry Measurement
        breakout_asymmetry = {}
        
        for period in periods:
            recent_data = current_data.iloc[-period:]
            compression_high = recent_data['high'].max()
            compression_low = recent_data['low'].min()
            current_close = current_data['close'].iloc[-1]
            current_range = compression_high - compression_low
            
            if current_range > 0:
                # Upward breakout strength
                upward_strength = (current_close - compression_low) / current_range
                # Downward breakout strength  
                downward_strength = (compression_high - current_close) / current_range
                # Asymmetry ratio
                breakout_asymmetry[period] = upward_strength - downward_strength
            else:
                breakout_asymmetry[period] = 0
        
        # Multi-fractal asymmetry convergence
        asymmetry_convergence = np.mean(list(breakout_asymmetry.values()))
        
        # 3. Volume Fractal Dynamics Integration
        current_volume = current_data['volume'].iloc[-1]
        
        # Volume concentration: current volume / 5-day volume sum
        volume_5day_sum = current_data['volume'].iloc[-5:].sum()
        volume_concentration = current_volume / volume_5day_sum if volume_5day_sum > 0 else 0
        
        # Volume persistence: current volume / previous volume
        if i > 0:
            prev_volume = current_data['volume'].iloc[-2]
            volume_persistence = current_volume / prev_volume if prev_volume > 0 else 1
        else:
            volume_persistence = 1
        
        # Volume fractal skew (simplified)
        recent_volumes = current_data['volume'].iloc[-5:]
        volume_skew = np.mean(recent_volumes.pct_change().dropna()) if len(recent_volumes) > 1 else 0
        
        # Trade size fractal: amount/volume
        trade_size = current_data['amount'].iloc[-1] / current_volume if current_volume > 0 else 0
        
        # 4. Composite Fractal Breakout Alpha
        # Combine fractal asymmetry with volume skew
        asymmetry_volume_component = asymmetry_convergence * (1 + volume_skew)
        
        # Weight by fractal compression duration (simplified)
        compression_weight = 1 + compression_alignment
        
        # Apply multi-timeframe breakout alignment
        breakout_alignment = np.std(list(breakout_asymmetry.values()))
        
        # Enhance with trade flow concentration
        flow_enhancement = 1 + volume_concentration * trade_size
        
        # Fractal directional bias
        directional_bias = np.sign(asymmetry_convergence) * (1 + abs(asymmetry_convergence))
        
        # Integrated fractal breakout factor
        fractal_factor = (asymmetry_volume_component * compression_weight * 
                         flow_enhancement * directional_bias * (1 - breakout_alignment))
        
        result.iloc[i] = fractal_factor
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
