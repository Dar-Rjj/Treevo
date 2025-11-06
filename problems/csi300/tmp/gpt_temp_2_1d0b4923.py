import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-scale Momentum-Volume Fractal Divergence Alpha Factor
    Combines price momentum acceleration with volume fractal patterns across multiple timeframes
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required minimum data length
    min_periods = 20
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # 1. Multi-period Fractal Momentum Calculation
        # Multi-scale Price Returns
        if i >= 5:
            short_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-5]) - 1
        else:
            short_return = 0
            
        if i >= 10:
            medium_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-10]) - 1
        else:
            medium_return = 0
            
        if i >= 20:
            long_return = (current_data['close'].iloc[i] / current_data['close'].iloc[i-20]) - 1
        else:
            long_return = 0
        
        # Fractal Momentum Acceleration
        if medium_return != 0 and abs(medium_return) > 1e-8:
            short_medium_accel = short_return / medium_return if abs(medium_return) > 1e-8 else 0
        else:
            short_medium_accel = 0
            
        if long_return != 0 and abs(long_return) > 1e-8:
            medium_long_accel = medium_return / long_return if abs(long_return) > 1e-8 else 0
        else:
            medium_long_accel = 0
        
        # Acceleration regime detection
        momentum_accelerating = (short_medium_accel > 1.0) and (medium_long_accel > 1.0)
        momentum_decelerating = (short_medium_accel < 1.0) and (medium_long_accel < 1.0)
        
        # 2. Volume Fractal Patterns Analysis
        # Multi-scale Volume Changes
        if i >= 5:
            volume_5d_change = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-5:i+1].mean()) - 1
        else:
            volume_5d_change = 0
            
        if i >= 10:
            volume_10d_change = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-10:i+1].mean()) - 1
        else:
            volume_10d_change = 0
            
        if i >= 20:
            volume_20d_change = (current_data['volume'].iloc[i] / current_data['volume'].iloc[i-20:i+1].mean()) - 1
        else:
            volume_20d_change = 0
        
        # Volume clustering regimes
        volume_burst_5d = volume_5d_change > 0.5  # 50% above average
        volume_burst_10d = volume_10d_change > 0.3  # 30% above average
        
        # Volume persistence (recent volume trend)
        if i >= 5:
            recent_volume_trend = np.corrcoef(range(5), current_data['volume'].iloc[i-4:i+1])[0,1] if len(current_data['volume'].iloc[i-4:i+1]) > 1 else 0
        else:
            recent_volume_trend = 0
        
        # 3. Fractal Divergence Signals
        # Price-Volume Momentum Divergence
        price_momentum_strength = (abs(short_return) + abs(medium_return) + abs(long_return)) / 3
        
        # Volume momentum strength
        volume_momentum_strength = (abs(volume_5d_change) + abs(volume_10d_change) + abs(volume_20d_change)) / 3
        
        # Divergence strength at each timeframe
        short_divergence = np.sign(short_return) != np.sign(volume_5d_change)
        medium_divergence = np.sign(medium_return) != np.sign(volume_10d_change)
        long_divergence = np.sign(long_return) != np.sign(volume_20d_change)
        
        divergence_count = sum([short_divergence, medium_divergence, long_divergence])
        
        # Multi-scale confirmation analysis
        volume_confirms_acceleration = (momentum_accelerating and volume_burst_5d and recent_volume_trend > 0)
        volume_confirms_deceleration = (momentum_decelerating and volume_5d_change < -0.2 and recent_volume_trend < 0)
        
        # 4. Composite Alpha Signal Generation
        base_signal = 0.0
        
        # Strong signal: Aligned acceleration with volume confirmation
        if momentum_accelerating and volume_confirms_acceleration:
            base_signal += 2.0
        # Divergence signal: Momentum-volume mismatch
        elif divergence_count >= 2:
            base_signal -= 1.5
        # Weak signal: Mixed patterns
        else:
            base_signal += 0.5 if price_momentum_strength > 0.02 else -0.5
        
        # Multi-scale signal integration
        timeframe_weights = [0.4, 0.35, 0.25]  # Short, medium, long weights
        timeframe_signals = [
            short_return * (1 if not short_divergence else -1),
            medium_return * (1 if not medium_divergence else -1),
            long_return * (1 if not long_divergence else -1)
        ]
        
        weighted_signal = sum(w * s for w, s in zip(timeframe_weights, timeframe_signals))
        
        # Pattern persistence assessment
        if i >= 10:
            # Recent price volatility as confidence measure
            recent_volatility = current_data['close'].iloc[i-9:i+1].pct_change().std()
            confidence_multiplier = 1.0 / (1.0 + recent_volatility * 10) if recent_volatility > 0 else 1.0
        else:
            confidence_multiplier = 1.0
        
        # Final composite signal
        composite_signal = base_signal + weighted_signal * 10
        final_signal = composite_signal * confidence_multiplier
        
        result.iloc[i] = final_signal
    
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
