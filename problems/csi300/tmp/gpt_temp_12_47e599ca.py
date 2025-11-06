import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Efficiency Components
    data['Price_Efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['Volume_Efficiency'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Asymmetric Price Dynamics
    data['Upside_Acceleration'] = (data['high'] - data['high'].shift(1)) - (data['high'].shift(1) - data['high'].shift(2))
    data['Downside_Acceleration'] = (data['low'] - data['low'].shift(1)) - (data['low'].shift(1) - data['low'].shift(2))
    data['Asymmetry_Ratio'] = data['Upside_Acceleration'] / (np.abs(data['Downside_Acceleration']) + 0.001)
    
    # Multi-Timeframe Momentum Structure
    # Short-term Momentum (3-day)
    data['Short_Price_Momentum'] = (data['close'] / data['close'].shift(3) - 1) * data['Price_Efficiency']
    data['Short_Volume_Momentum'] = (data['volume'] / data['volume'].shift(3) - 1) * data['Volume_Efficiency']
    
    # Medium-term Momentum (10-day)
    data['Medium_Price_Momentum'] = (data['close'] / data['close'].shift(10) - 1) * data['Price_Efficiency']
    data['Medium_Volume_Momentum'] = (data['volume'] / data['volume'].shift(10) - 1) * data['Volume_Efficiency']
    
    # Momentum Divergence
    data['Price_Divergence'] = data['Short_Price_Momentum'] - data['Medium_Price_Momentum']
    data['Volume_Divergence'] = data['Short_Volume_Momentum'] - data['Medium_Volume_Momentum']
    
    # Fractal Efficiency Patterns
    price_changes = np.abs(data['close'].diff())
    data['Price_Fractal_Efficiency'] = np.abs(data['close'] - data['close'].shift(10)) / (
        price_changes.rolling(window=10, min_periods=1).sum() + 1e-8)
    data['Volume_Fractal_Pattern'] = (data['volume'] / data['volume'].shift(3)) - (data['volume'] / data['volume'].shift(10))
    
    # Volume-Weighted Acceleration
    data['Volume_Acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    data['Volume_Persistence'] = data['volume'] / data['volume'].shift(5)
    data['Upside_Volume_Factor'] = data['Upside_Acceleration'] * data['Volume_Persistence']
    data['Downside_Volume_Factor'] = data['Downside_Acceleration'] * data['Volume_Acceleration']
    data['Net_Volume_Bias'] = data['Upside_Volume_Factor'] - data['Downside_Volume_Factor']
    
    # Range-Based Pressure Signals
    range_pressure = []
    for i in range(len(data)):
        if i >= 2:
            pressure_sum = 0
            for j in range(3):
                idx = i - j
                high_low_range = data['high'].iloc[idx] - data['low'].iloc[idx]
                if high_low_range > 1e-8:
                    pressure_sum += (data['close'].iloc[idx] - data['low'].iloc[idx]) / high_low_range
            range_pressure.append(pressure_sum)
        else:
            range_pressure.append(np.nan)
    data['Range_Pressure'] = range_pressure
    data['Amount_Momentum'] = data['amount'] / data['amount'].shift(3) - 1
    
    # Regime Classification
    data['High_Efficiency'] = data['Price_Fractal_Efficiency'] > 0.6
    data['Low_Efficiency'] = data['Price_Fractal_Efficiency'] < 0.4
    data['Transition'] = data['Volume_Acceleration'] > 2
    
    divergence_sign = np.sign(data['Price_Divergence']) * np.sign(data['Volume_Divergence'])
    data['Consistent_Divergence'] = divergence_sign > 0
    data['Conflicting_Divergence'] = divergence_sign < 0
    
    # Adaptive Signal Integration
    # Core Momentum Signal
    data['Asymmetric_Momentum'] = (data['Short_Price_Momentum'] + data['Medium_Price_Momentum']) * data['Asymmetry_Ratio']
    data['Volume_Confirmed_Momentum'] = data['Asymmetric_Momentum'] * data['Net_Volume_Bias']
    
    # Efficiency-Enhanced Signals
    efficiency_enhanced = np.zeros(len(data))
    for i in range(len(data)):
        if data['High_Efficiency'].iloc[i]:
            efficiency_enhanced[i] = (data['Volume_Confirmed_Momentum'].iloc[i] * 
                                    data['Price_Fractal_Efficiency'].iloc[i] * 
                                    data['Range_Pressure'].iloc[i])
        elif data['Low_Efficiency'].iloc[i]:
            efficiency_enhanced[i] = (data['Volume_Confirmed_Momentum'].iloc[i] * 
                                    data['Volume_Fractal_Pattern'].iloc[i] * 
                                    data['Amount_Momentum'].iloc[i])
        elif data['Transition'].iloc[i]:
            efficiency_enhanced[i] = (data['Volume_Confirmed_Momentum'].iloc[i] * 
                                    data['Price_Divergence'].iloc[i] * 
                                    data['Volume_Acceleration'].iloc[i])
        else:
            efficiency_enhanced[i] = data['Volume_Confirmed_Momentum'].iloc[i]
    
    data['Efficiency_Enhanced_Signal'] = efficiency_enhanced
    
    # Final Alpha Factor
    divergence_weighted = []
    for i in range(len(data)):
        if data['Consistent_Divergence'].iloc[i]:
            divergence_weighted.append(data['Efficiency_Enhanced_Signal'].iloc[i] * 1.2)
        elif data['Conflicting_Divergence'].iloc[i]:
            divergence_weighted.append(data['Efficiency_Enhanced_Signal'].iloc[i] * 0.8)
        else:
            divergence_weighted.append(data['Efficiency_Enhanced_Signal'].iloc[i])
    
    data['Divergence_Weighted_Output'] = divergence_weighted
    
    # Final Alpha with Volume Acceleration sign
    alpha_factor = data['Divergence_Weighted_Output'] * np.sign(data['Volume_Acceleration'])
    
    return alpha_factor
