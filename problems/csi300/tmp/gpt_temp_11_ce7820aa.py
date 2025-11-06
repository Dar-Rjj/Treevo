import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Fractal Pressure Alignment alpha factor
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Required window for calculations
    window = 20
    
    for i in range(window, len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Fractal Efficiency Assessment
        # Hurst exponent estimation (20-day close price series)
        close_window = current_data['close'].iloc[-window:]
        if len(close_window) < window:
            continue
            
        # Calculate Hurst exponent using R/S method
        lags = range(2, min(10, len(close_window)//2))
        rs_values = []
        
        for lag in lags:
            # Split series into non-overlapping windows
            n_windows = len(close_window) // lag
            if n_windows < 2:
                continue
                
            rs_window = []
            for j in range(n_windows):
                segment = close_window.iloc[j*lag:(j+1)*lag]
                if len(segment) < 2:
                    continue
                    
                # Calculate R/S for this segment
                mean_val = segment.mean()
                deviations = segment - mean_val
                cumulative_dev = deviations.cumsum()
                r = cumulative_dev.max() - cumulative_dev.min()
                s = segment.std()
                
                if s > 0:
                    rs_window.append(r / s)
            
            if rs_window:
                rs_values.append(np.log(np.mean(rs_window)))
        
        if len(rs_values) >= 2:
            hurst_exponent = np.polyfit(np.log(list(lags)[:len(rs_values)]), rs_values, 1)[0]
        else:
            hurst_exponent = 0.5
        
        # Fractal dimension calculation using high-low range
        high_low_range = current_data['high'].iloc[-window:] - current_data['low'].iloc[-window:]
        if high_low_range.std() > 0:
            # Simplified fractal dimension estimation
            log_range = np.log(high_low_range + 1e-8)
            log_time = np.log(np.arange(1, len(high_low_range) + 1))
            fractal_dim = 2 - np.polyfit(log_time, log_range, 1)[0]
        else:
            fractal_dim = 1.5
        
        # 2. Volatility Regime Classification
        if i >= 4:
            # Volatility Ratio
            current_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
            prev_4day_range = current_data['high'].iloc[-5:-1] - current_data['low'].iloc[-5:-1]
            avg_prev_range = prev_4day_range.mean() if len(prev_4day_range) > 0 else current_range
            
            if avg_prev_range > 0:
                volatility_ratio = current_range / avg_prev_range
            else:
                volatility_ratio = 1.0
            
            # Volatility Acceleration
            if i >= 5:
                prev_range = current_data['high'].iloc[-2] - current_data['low'].iloc[-2]
                prev_prev_4day_range = current_data['high'].iloc[-6:-2] - current_data['low'].iloc[-6:-2]
                avg_prev_prev_range = prev_prev_4day_range.mean() if len(prev_prev_4day_range) > 0 else prev_range
                
                if avg_prev_prev_range > 0:
                    prev_volatility_ratio = prev_range / avg_prev_prev_range
                    volatility_acceleration = volatility_ratio / prev_volatility_ratio if prev_volatility_ratio > 0 else 1.0
                else:
                    volatility_acceleration = 1.0
            else:
                volatility_acceleration = 1.0
            
            # Regime flags
            high_vol_flag = volatility_ratio > 1.5
            low_vol_flag = volatility_ratio < 0.7
        else:
            volatility_ratio = 1.0
            volatility_acceleration = 1.0
            high_vol_flag = False
            low_vol_flag = False
        
        # 3. Pressure Accumulation Measurement
        if i >= 4:
            # Daily Pressure
            daily_range = current_data['high'].iloc[-1] - current_data['low'].iloc[-1]
            if daily_range > 0:
                daily_pressure = (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) / daily_range - 0.5
            else:
                daily_pressure = 0.0
            
            # Cumulative Pressure (last 5 days)
            cumulative_pressure = 0.0
            for j in range(max(0, i-4), i+1):
                day_range = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                if day_range > 0:
                    day_pressure = (current_data['close'].iloc[j] - current_data['low'].iloc[j]) / day_range - 0.5
                    cumulative_pressure += day_pressure
            
            # Pressure divergence detection
            pressure_extremes = abs(cumulative_pressure) > 1.0
            
            # Pressure Trend
            if i >= 6:
                prev_cumulative = 0.0
                for j in range(max(0, i-6), i-1):
                    day_range = current_data['high'].iloc[j] - current_data['low'].iloc[j]
                    if day_range > 0:
                        day_pressure = (current_data['close'].iloc[j] - current_data['low'].iloc[j]) / day_range - 0.5
                        prev_cumulative += day_pressure
                
                pressure_trend = np.sign(cumulative_pressure - prev_cumulative)
            else:
                pressure_trend = 0
        else:
            cumulative_pressure = 0.0
            pressure_extremes = False
            pressure_trend = 0
        
        # 4. Multi-timeframe Alignment Analysis
        high_efficiency = hurst_exponent > 0.6
        low_efficiency = hurst_exponent < 0.4
        
        pressure_efficiency_alignment = np.sign(cumulative_pressure) * (hurst_exponent - 0.5) if cumulative_pressure != 0 else 0
        convergence_magnitude = abs(cumulative_pressure) * abs(hurst_exponent - 0.5)
        
        # 5. Alpha Signal Generation
        alpha_signal = 0.0
        
        # Efficient Pressure Release
        if high_efficiency and pressure_extremes and volatility_acceleration > 1.1:
            alpha_signal += convergence_magnitude * np.sign(cumulative_pressure)
        
        # Fractal Breakdown Anticipation
        if low_efficiency and pressure_trend != np.sign(cumulative_pressure) and high_vol_flag and cumulative_pressure != 0:
            alpha_signal -= convergence_magnitude * 0.8
        
        # Regime Transition Signal
        if i >= 5 and pressure_efficiency_alignment > 0:
            prev_vol_ratio = current_data['high'].iloc[-2] - current_data['low'].iloc[-2]
            prev_prev_4day_range = current_data['high'].iloc[-6:-2] - current_data['low'].iloc[-6:-2]
            avg_prev_prev_range = prev_prev_4day_range.mean() if len(prev_prev_4day_range) > 0 else prev_vol_ratio
            
            if avg_prev_prev_range > 0:
                prev_volatility_ratio = prev_vol_ratio / avg_prev_prev_range
                # Check if crossing 1.0 threshold
                if (prev_volatility_ratio < 1.0 and volatility_ratio >= 1.0) or (prev_volatility_ratio >= 1.0 and volatility_ratio < 1.0):
                    alpha_signal += pressure_efficiency_alignment * 0.5
        
        result.iloc[i] = alpha_signal
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
