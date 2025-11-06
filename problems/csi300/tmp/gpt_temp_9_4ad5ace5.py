import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Fracture Detection System combined with Pressure Accumulation Divergence
    and Range Expansion Quality to generate alpha signals.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    # Required minimum data points
    min_periods = 20
    
    for i in range(min_periods, len(data)):
        current_data = data.iloc[:i+1].copy()
        
        # 1. Momentum Fracture Detection
        # Gap Opening Analysis
        overnight_gap = (current_data['open'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]
        recent_volatility = current_data['close'].iloc[-min_periods:].pct_change().std()
        gap_vs_volatility = abs(overnight_gap) / (recent_volatility + 1e-8)
        
        # Intraday Fracture Signals
        daily_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
        close_position = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / (daily_range + 1e-8)
        
        # High-Low Breakthrough Failure
        prev_high = current_data['high'].iloc[-2]
        prev_low = current_data['low'].iloc[-2]
        high_break_failure = 1 if (current_data['high'].iloc[-1] > prev_high and current_data['close'].iloc[-1] < prev_high) else 0
        low_break_failure = 1 if (current_data['low'].iloc[-1] < prev_low and current_data['close'].iloc[-1] > prev_low) else 0
        
        # Volume Fracture Detection
        volume_ma = current_data['volume'].iloc[-min_periods:].mean()
        volume_spike = current_data['volume'].iloc[-1] / (volume_ma + 1e-8)
        
        # Volume-Price Divergence
        price_change = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['close'].iloc[-2]
        volume_price_divergence = volume_spike / (price_change + 1e-8)
        
        # Calculate Fracture Severity Score
        price_fracture_severity = (
            gap_vs_volatility * 0.3 +
            (1 - abs(close_position - 0.5)) * 0.2 +  # Extreme close positions indicate fracture
            (high_break_failure + low_break_failure) * 0.25 +
            abs(overnight_gap) * 0.25
        )
        
        volume_fracture_severity = (
            min(volume_spike, 3) * 0.4 +  # Cap extreme spikes
            min(volume_price_divergence, 5) * 0.3 +  # Cap extreme divergence
            (1 if volume_spike > 2 and price_change < 0.01 else 0) * 0.3
        )
        
        momentum_fracture_score = price_fracture_severity * 0.6 + volume_fracture_severity * 0.4
        
        # 2. Pressure Accumulation Divergence
        # Buying/Selling Pressure Accumulation
        pressure_direction = 1 if current_data['close'].iloc[-1] > current_data['open'].iloc[-1] else -1
        pressure_strength = abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / current_data['open'].iloc[-1]
        
        # Multi-day Pressure Patterns
        recent_closes = current_data['close'].iloc[-5:]
        high_close_streak = sum((recent_closes > recent_closes.shift(1)).dropna())
        low_close_streak = sum((recent_closes < recent_closes.shift(1)).dropna())
        
        # Pressure Divergence
        price_move = current_data['close'].iloc[-1] - current_data['close'].iloc[-2]
        normalized_price_move = price_move / (current_data['close'].iloc[-2] + 1e-8)
        
        pressure_divergence = (
            pressure_strength * pressure_direction - 
            normalized_price_move * (1 if normalized_price_move > 0 else -1)
        )
        
        # 3. Range Expansion Quality Indicator
        current_range = daily_range / current_data['close'].iloc[-1]
        avg_range = (current_data['high'].iloc[-min_periods:] - current_data['low'].iloc[-min_periods:]).mean() / current_data['close'].iloc[-min_periods:].mean()
        
        range_expansion = current_range / (avg_range + 1e-8)
        
        # Expansion Direction Quality
        if close_position > 0.6:  # Strong upward expansion
            direction_quality = close_position * range_expansion
        elif close_position < 0.4:  # Strong downward expansion
            direction_quality = (1 - close_position) * range_expansion
        else:  # Neutral/weak expansion
            direction_quality = 0
        
        # Volume concentration analysis
        upper_volume_quality = 1 if (close_position > 0.7 and volume_spike > 1.5) else 0
        lower_volume_quality = 1 if (close_position < 0.3 and volume_spike > 1.5) else 0
        
        range_quality_score = (
            direction_quality * 0.5 +
            (upper_volume_quality - lower_volume_quality) * 0.3 +
            min(range_expansion, 3) * 0.2  # Cap extreme expansions
        )
        
        # 4. Combine all components
        # Momentum fracture gives reversal signals (negative weight)
        # Pressure accumulation gives continuation signals (positive weight)
        # Range quality gives confirmation signals (positive weight for quality moves)
        
        final_signal = (
            -momentum_fracture_score * 0.4 +  # Fractures suggest reversals
            pressure_divergence * 2.0 +  # Pressure suggests continuation
            range_quality_score * 0.6  # Quality confirms direction
        )
        
        # Add trend context
        short_trend = current_data['close'].iloc[-1] > current_data['close'].iloc[-5]
        medium_trend = current_data['close'].iloc[-1] > current_data['close'].iloc[-10]
        
        trend_strength = (1 if short_trend else -1) * 0.3 + (1 if medium_trend else -1) * 0.7
        
        # Final alpha signal with trend adjustment
        alpha_signal.iloc[i] = final_signal * (1 + 0.2 * trend_strength)
    
    # Fill initial values with 0
    alpha_signal = alpha_signal.fillna(0)
    
    return alpha_signal
