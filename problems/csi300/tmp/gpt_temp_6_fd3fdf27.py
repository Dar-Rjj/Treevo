import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Fractal Velocity-Pressure Dynamics
    # Fractal velocity pressure
    close_shift_5 = data['close'].shift(5)
    close_shift_10 = data['close'].shift(10)
    volume_shift_1 = data['volume'].shift(1)
    
    fractal_velocity_pressure = ((data['close'] - close_shift_5) / 
                               (close_shift_5 - close_shift_10 + 1e-8) * 
                               (data['high'] - data['open']) - 
                               (data['open'] - data['low']) * 
                               data['volume'] / (volume_shift_1 + 1e-8))
    
    # Fractal pressure shifts
    high_shift_1 = data['high'].shift(1)
    low_shift_1 = data['low'].shift(1)
    
    fractal_pressure_shifts = ((data['high'] - data['open']) - 
                             (data['open'] - data['low']) * 
                             (data['high'] - data['low']) / 
                             ((high_shift_1 - low_shift_1) + 1e-8) * 
                             data['volume'] / (volume_shift_1 + 1e-8))
    
    # Volume fractal pressure
    volume_shift_5 = data['volume'].shift(5)
    volume_shift_10 = data['volume'].shift(10)
    
    volume_fractal_pressure = ((data['volume'] / (volume_shift_5 + 1e-8)) / 
                             (volume_shift_5 / (volume_shift_10 + 1e-8)) * 
                             (data['volume'] / ((data['high'] - data['low']) + 1e-8)) - 
                             (volume_shift_1 / ((high_shift_1 - low_shift_1) + 1e-8)) * 
                             (data['close'] - data['open']) / (data['volume'] + 1e-8))
    
    # Multi-Timeframe Pressure Validation
    # Fractal pressure consistency
    close_shift_1 = data['close'].shift(1)
    close_shift_2 = data['close'].shift(2)
    
    sign_consistency = []
    for i in range(len(data)):
        if i >= 2:
            signs = []
            for j in range(i-2, i+1):
                if j >= 1:
                    sign_current = np.sign(data['close'].iloc[j] - data['close'].iloc[j-1])
                    sign_prev = np.sign(data['close'].iloc[j-1] - data['close'].iloc[j-2]) if j >= 2 else 0
                    signs.append(1 if sign_current == sign_prev else 0)
            sign_consistency.append(sum(signs) / 3 if len(signs) > 0 else 0)
        else:
            sign_consistency.append(0)
    
    fractal_pressure_consistency = pd.Series(sign_consistency, index=data.index) * data['volume'] / (volume_shift_1 + 1e-8)
    
    # Fractal pressure efficiency
    efficiency_consistency = []
    for i in range(len(data)):
        if i >= 2:
            signs = []
            for j in range(i-2, i+1):
                if j >= 2:
                    range_ratio_current = abs(data['close'].iloc[j] - data['open'].iloc[j]) / ((data['high'].iloc[j] - data['low'].iloc[j]) + 1e-8)
                    range_ratio_prev = abs(data['close'].iloc[j-1] - data['open'].iloc[j-1]) / ((data['high'].iloc[j-1] - data['low'].iloc[j-1]) + 1e-8)
                    range_ratio_prev2 = abs(data['close'].iloc[j-2] - data['open'].iloc[j-2]) / ((data['high'].iloc[j-2] - data['low'].iloc[j-2]) + 1e-8)
                    
                    sign_current = np.sign(range_ratio_current - range_ratio_prev)
                    sign_prev = np.sign(range_ratio_prev - range_ratio_prev2)
                    signs.append(1 if sign_current == sign_prev else 0)
            efficiency_consistency.append(sum(signs) / 3 if len(signs) > 0 else 0)
        else:
            efficiency_consistency.append(0)
    
    fractal_pressure_efficiency = pd.Series(efficiency_consistency, index=data.index) * data['volume'] / (volume_shift_1 + 1e-8)
    
    # Fractal pressure regime
    regime_counts = []
    for i in range(len(data)):
        if i >= 4:
            count = 0
            for j in range(i-4, i+1):
                if j >= 1:
                    range_current = data['high'].iloc[j] - data['low'].iloc[j]
                    range_prev = data['high'].iloc[j-1] - data['low'].iloc[j-1]
                    if range_current / (range_prev + 1e-8) > 1.2:
                        count += 1
            regime_counts.append(count / 5)
        else:
            regime_counts.append(0)
    
    fractal_pressure_regime = pd.Series(regime_counts, index=data.index) * data['volume'] / (volume_shift_1 + 1e-8)
    
    # Cross-Session Pressure Dynamics
    close_shift_1 = data['close'].shift(1)
    high_shift_2 = data['high'].shift(2)
    low_shift_2 = data['low'].shift(2)
    
    # Gap pressure momentum
    gap_pressure_momentum = ((data['open'] - close_shift_1) / (data['amount'] + 1e-8) * 
                           np.sign(data['close'] - data['open']) * 
                           (high_shift_2 - low_shift_2))
    
    # Intraday pressure absorption
    intraday_pressure_absorption = ((data['high'] - data['low']) / (data['amount'] + 1e-8) * 
                                  (high_shift_2 - low_shift_2))
    
    # Session pressure persistence
    amount_shift_1 = data['amount'].shift(1)
    session_pressure_persistence = (((data['close'] - data['open']) / (data['amount'] + 1e-8) - 
                                   (close_shift_1 - data['open'].shift(1)) / (amount_shift_1 + 1e-8)) * 
                                   (high_shift_2 - low_shift_2))
    
    # Volatility-Pressure Coupling
    # Fractal range expansion
    fractal_range_expansion = ((data['high'] - data['low']) / ((high_shift_1 - low_shift_1) + 1e-8) * 
                             data['volume'] / (volume_shift_1 + 1e-8))
    
    # Fractal pressure absorption
    fractal_pressure_absorption = (data['volume'] / ((data['high'] - data['low']) + 1e-8) * 
                                 fractal_pressure_efficiency)
    
    # Fractal pressure breakout
    close_shift_3 = data['close'].shift(3)
    volume_shift_3 = data['volume'].shift(3)
    
    rolling_max = data['close'].shift(1).rolling(window=3, min_periods=1).max()
    fractal_pressure_breakout = ((data['close'] - rolling_max) / (close_shift_3 + 1e-8) * 
                               data['volume'] / (volume_shift_3 + 1e-8))
    
    # Price-Pressure Fractal Alignment
    close_shift_2 = data['close'].shift(2)
    efficiency_shift_2 = fractal_pressure_efficiency.shift(2)
    
    # Price-pressure divergence
    price_pressure_divergence = ((data['close'] / (close_shift_2 + 1e-8) - 1) - 
                               (fractal_pressure_efficiency / (efficiency_shift_2 + 1e-8) - 1))
    
    # Fractal pressure persistence
    volume_shift_1 = data['volume'].shift(1)
    efficiency_shift_1 = fractal_pressure_efficiency.shift(1)
    
    fractal_pressure_persistence = (np.sign(data['close'] - close_shift_1) * 
                                  np.sign(data['volume'] - volume_shift_1) * 
                                  np.sign(fractal_pressure_efficiency - efficiency_shift_1))
    
    # Fractal pressure quality
    close_shift_3 = data['close'].shift(3)
    fractal_pressure_quality = ((data['close'] / (close_shift_3 + 1e-8) - 1) * 
                              fractal_pressure_persistence)
    
    # Fractal Pressure-Adaptive Alpha
    # Short-term pressure
    short_term_pressure = (fractal_velocity_pressure * 
                         fractal_pressure_persistence * 
                         gap_pressure_momentum)
    
    # Medium-term pressure
    medium_term_pressure = (fractal_pressure_quality * 
                          fractal_pressure_absorption * 
                          session_pressure_persistence)
    
    # Long-term pressure
    long_term_pressure = (fractal_pressure_regime * 
                        price_pressure_divergence * 
                        fractal_pressure_breakout)
    
    # Final alpha factor - weighted combination
    alpha_factor = (0.4 * short_term_pressure + 
                   0.35 * medium_term_pressure + 
                   0.25 * long_term_pressure)
    
    # Clean infinite and NaN values
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_factor
