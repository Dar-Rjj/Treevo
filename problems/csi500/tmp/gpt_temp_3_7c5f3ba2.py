import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Confirmed Multi-Timeframe Momentum Factor with Regime Adaptation
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Minimum required data points
    min_periods = 20
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1]
        
        # Multi-Timeframe Momentum Calculation
        # Ultra-short momentum (1-2 days)
        ultra_short_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-2] - 1) * 100
        
        # Short-term momentum (3-5 days) using slope
        if i >= 5:
            short_prices = current_data['close'].iloc[-5:]
            short_slope = np.polyfit(range(5), short_prices.values, 1)[0] / short_prices.mean() * 100
        else:
            short_slope = ultra_short_momentum
        
        # Medium-term momentum (10-15 days) using ROC
        if i >= 15:
            medium_roc = (current_data['close'].iloc[-1] / current_data['close'].iloc[-15] - 1) * 100
        else:
            medium_roc = short_slope
        
        # Momentum acceleration
        momentum_acceleration = short_slope - medium_roc
        
        # Volume Confirmation Filter
        # Volume trend analysis (5-day volume slope)
        if i >= 5:
            volume_5d = current_data['volume'].iloc[-5:]
            volume_slope = np.polyfit(range(5), volume_5d.values, 1)[0] / volume_5d.mean() * 100
        else:
            volume_slope = 0
        
        # Volume anomaly detection (current vs 20-day average)
        if i >= 20:
            volume_20d_avg = current_data['volume'].iloc[-20:].mean()
            volume_anomaly = (current_data['volume'].iloc[-1] / volume_20d_avg - 1) * 100
        else:
            volume_anomaly = 0
        
        # Volume-momentum alignment scoring
        volume_momentum_score = 0
        if momentum_acceleration > 0 and volume_slope > 0:
            volume_momentum_score = min(momentum_acceleration, volume_slope) / 100
        elif momentum_acceleration < 0 and volume_slope < 0:
            volume_momentum_score = max(momentum_acceleration, volume_slope) / 100
        
        # Volume confirmation weight
        volume_confirmation = 0.3 * np.sign(volume_slope) + 0.4 * np.tanh(volume_anomaly/50) + 0.3 * volume_momentum_score
        
        # Regime-Adaptive Components
        # Volatility regime using true range expansion
        if i >= 5:
            true_ranges = []
            for j in range(1, 6):
                if i-j >= 0:
                    high_low = current_data['high'].iloc[-j] - current_data['low'].iloc[-j]
                    high_close = abs(current_data['high'].iloc[-j] - current_data['close'].iloc[-j-1])
                    low_close = abs(current_data['low'].iloc[-j] - current_data['close'].iloc[-j-1])
                    true_ranges.append(max(high_low, high_close, low_close))
            
            avg_true_range = np.mean(true_ranges) if true_ranges else 0
            volatility_regime = avg_true_range / current_data['close'].iloc[-1]
        else:
            volatility_regime = 0.01
        
        # Dynamic timeframe adjustment based on volatility
        if volatility_regime > 0.02:  # High volatility
            momentum_weight = 0.6
            volume_weight = 0.4
        else:  # Low volatility
            momentum_weight = 0.4
            volume_weight = 0.6
        
        # Breakout/Reversal Detection
        # Price position in recent range
        if i >= 10:
            recent_high = current_data['high'].iloc[-10:].max()
            recent_low = current_data['low'].iloc[-10:].min()
            price_position = (current_data['close'].iloc[-1] - recent_low) / (recent_high - recent_low)
        else:
            price_position = 0.5
        
        # Volume conviction for breakouts
        volume_conviction = 0
        if price_position > 0.8 and volume_anomaly > 20:  # Potential breakout
            volume_conviction = min(1.0, volume_anomaly / 100)
        elif price_position < 0.2 and volume_anomaly > 20:  # Potential reversal
            volume_conviction = -min(1.0, volume_anomaly / 100)
        
        # Composite factor calculation
        momentum_component = momentum_acceleration * momentum_weight
        volume_component = volume_confirmation * volume_weight * 100
        breakout_component = volume_conviction * 50
        
        # Final factor value
        factor_value = (momentum_component + volume_component + breakout_component) / 3
        
        result.iloc[i] = factor_value
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
