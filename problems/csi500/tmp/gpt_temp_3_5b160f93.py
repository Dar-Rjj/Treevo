import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate integrated multi-timeframe alpha factor combining:
    - Volatility-scaled momentum across 3, 5, 10-day timeframes
    - Volume-price divergence with intraday confirmation
    - Multi-timeframe intraday convergence patterns
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required lookback for calculations
    lookback = 10
    
    for i in range(lookback, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Volatility-Scaled Multi-Timeframe Momentum
        momentum_components = []
        vol_components = []
        
        # Momentum and volatility for 3, 5, 10-day timeframes
        for period in [3, 5, 10]:
            if i >= period:
                # Momentum calculation
                momentum = (current_data['close'].iloc[i] / current_data['close'].iloc[i-period] - 1)
                momentum_components.append(momentum)
                
                # Volatility calculation (std of returns over the period)
                returns = []
                for j in range(period):
                    if i - period - j >= 0:
                        ret = (current_data['close'].iloc[i-j] / current_data['close'].iloc[i-period-j] - 1)
                        returns.append(ret)
                
                if len(returns) > 1:
                    vol = np.std(returns)
                    vol_components.append(vol if vol > 0 else 1e-6)
                else:
                    vol_components.append(1e-6)
            else:
                momentum_components.append(0)
                vol_components.append(1e-6)
        
        # Volatility-scaled momentum
        vol_scaled_momentum = []
        for mom, vol in zip(momentum_components, vol_components):
            vol_scaled_momentum.append(mom / vol)
        
        # Geometric mean of volatility-scaled momentum
        vol_momentum_signal = np.prod([abs(x) for x in vol_scaled_momentum]) ** (1/len(vol_scaled_momentum))
        vol_momentum_signal *= np.sign(np.prod(vol_scaled_momentum))
        
        # 2. Volume-Price Divergence with Intraday Confirmation
        # Price strength components
        if i >= 1:
            # Current day intraday positioning
            high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
            if high_low_range > 0:
                close_to_high_ratio = (current_data['close'].iloc[i] - current_data['low'].iloc[i]) / high_low_range
                open_to_close_strength = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range
            else:
                close_to_high_ratio = 0.5
                open_to_close_strength = 0
            
            # Previous day positioning
            prev_high_low_range = current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]
            if prev_high_low_range > 0:
                prev_close_to_high = (current_data['close'].iloc[i-1] - current_data['low'].iloc[i-1]) / prev_high_low_range
            else:
                prev_close_to_high = 0.5
            
            # Multi-period price strength
            price_strength = (close_to_high_ratio * open_to_close_strength) ** 0.5
            price_strength *= ((close_to_high_ratio + prev_close_to_high) / 2)
        
        else:
            price_strength = 0
        
        # Volume divergence analysis
        volume_ratios = []
        if i >= 1:
            # Short-term volume ratio
            if current_data['volume'].iloc[i-1] > 0:
                vol_ratio_1 = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1]
                volume_ratios.append(vol_ratio_1)
            
            # Medium-term volume ratio (5-day)
            if i >= 4:
                avg_vol_medium = current_data['volume'].iloc[i-4:i].mean()
                if avg_vol_medium > 0:
                    vol_ratio_5 = current_data['volume'].iloc[i] / avg_vol_medium
                    volume_ratios.append(vol_ratio_5)
            
            # Long-term volume ratio (10-day)
            if i >= 9:
                avg_vol_long = current_data['volume'].iloc[i-9:i].mean()
                if avg_vol_long > 0:
                    vol_ratio_10 = current_data['volume'].iloc[i] / avg_vol_long
                    volume_ratios.append(vol_ratio_10)
        
        if volume_ratios:
            volume_acceleration = np.prod(volume_ratios) ** (1/len(volume_ratios))
        else:
            volume_acceleration = 1
        
        # Combined volume-price divergence
        volume_price_signal = price_strength * volume_acceleration
        
        # 3. Multi-Timeframe Intraday Convergence
        # Session-based momentum
        high_low_range = current_data['high'].iloc[i] - current_data['low'].iloc[i]
        if high_low_range > 0:
            # Morning session analysis
            open_low_range = current_data['open'].iloc[i] - current_data['low'].iloc[i]
            if open_low_range > 0:
                morning_momentum = (current_data['high'].iloc[i] - current_data['open'].iloc[i]) / open_low_range
            else:
                morning_momentum = 1
            
            morning_efficiency = (current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range
            
            # Afternoon session analysis
            close_low_range = current_data['close'].iloc[i] - current_data['low'].iloc[i]
            high_close_range = current_data['high'].iloc[i] - current_data['close'].iloc[i]
            
            if close_low_range > 0 and high_close_range > 0:
                closing_momentum = close_low_range / high_close_range
                afternoon_pressure = high_close_range / close_low_range
            else:
                closing_momentum = 1
                afternoon_pressure = 1
            
            # Session alignment
            session_signal = (morning_momentum * morning_efficiency * closing_momentum * afternoon_pressure) ** 0.25
        
        else:
            session_signal = 1
        
        # Multi-period intraday patterns
        if i >= 1:
            # Previous day comparison
            prev_high_low_range = current_data['high'].iloc[i-1] - current_data['low'].iloc[i-1]
            if prev_high_low_range > 0:
                prev_morning_efficiency = (current_data['close'].iloc[i-1] - current_data['open'].iloc[i-1]) / prev_high_low_range
                
                prev_close_low_range = current_data['close'].iloc[i-1] - current_data['low'].iloc[i-1]
                prev_high_close_range = current_data['high'].iloc[i-1] - current_data['close'].iloc[i-1]
                
                if prev_close_low_range > 0 and prev_high_close_range > 0:
                    prev_closing_momentum = prev_close_low_range / prev_high_close_range
                else:
                    prev_closing_momentum = 1
                
                multi_period_signal = (session_signal * prev_morning_efficiency * prev_closing_momentum) ** (1/3)
            else:
                multi_period_signal = session_signal
        else:
            multi_period_signal = session_signal
        
        # Daily range efficiency scaling
        if high_low_range > 0:
            daily_range_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / high_low_range
        else:
            daily_range_efficiency = 0
        
        intraday_signal = multi_period_signal * daily_range_efficiency
        
        # 4. Integrated Multi-Timeframe Alpha
        # Combine all three components geometrically
        components = [vol_momentum_signal, volume_price_signal, intraday_signal]
        valid_components = [comp for comp in components if not np.isnan(comp) and not np.isinf(comp)]
        
        if valid_components:
            # Geometric mean with sign preservation
            abs_components = [abs(x) for x in valid_components]
            combined_signal = np.prod(abs_components) ** (1/len(valid_components))
            combined_signal *= np.sign(np.prod(valid_components))
        else:
            combined_signal = 0
        
        result.iloc[i] = combined_signal
    
    # Forward fill any remaining NaN values
    result = result.fillna(method='ffill').fillna(0)
    
    return result
