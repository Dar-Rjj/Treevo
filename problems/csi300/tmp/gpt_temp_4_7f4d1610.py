import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Fractal Structure
    # Opening Volatility Fractal
    data['Opening_Volatility_Strength'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 0.001)
    
    # Calculate consecutive days where Opening_Volatility_Strength > 0.6
    data['Opening_Volatility_Persistence'] = 0
    for i in range(1, len(data)):
        if data['Opening_Volatility_Strength'].iloc[i] > 0.6:
            if data['Opening_Volatility_Strength'].iloc[i-1] > 0.6:
                data['Opening_Volatility_Persistence'].iloc[i] = data['Opening_Volatility_Persistence'].iloc[i-1] + 1
            else:
                data['Opening_Volatility_Persistence'].iloc[i] = 1
        else:
            data['Opening_Volatility_Persistence'].iloc[i] = 0
    
    data['Opening_Volatility_Momentum'] = data['Opening_Volatility_Strength'] * data['Opening_Volatility_Persistence']
    
    # Closing Volatility Fractal
    data['Closing_Volatility_Strength'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)
    
    # Calculate consecutive days where Closing_Volatility_Strength > 0.6
    data['Closing_Volatility_Persistence'] = 0
    for i in range(1, len(data)):
        if data['Closing_Volatility_Strength'].iloc[i] > 0.6:
            if data['Closing_Volatility_Strength'].iloc[i-1] > 0.6:
                data['Closing_Volatility_Persistence'].iloc[i] = data['Closing_Volatility_Persistence'].iloc[i-1] + 1
            else:
                data['Closing_Volatility_Persistence'].iloc[i] = 1
        else:
            data['Closing_Volatility_Persistence'].iloc[i] = 0
    
    data['Closing_Volatility_Momentum'] = data['Closing_Volatility_Strength'] * data['Closing_Volatility_Persistence']
    
    # Volatility Pattern Divergence
    data['Volatility_Strength_Divergence'] = data['Opening_Volatility_Strength'] - data['Closing_Volatility_Strength']
    data['Volatility_Persistence_Divergence'] = data['Opening_Volatility_Persistence'] - data['Closing_Volatility_Persistence']
    data['Volatility_Momentum_Divergence'] = data['Opening_Volatility_Momentum'] - data['Closing_Volatility_Momentum']
    
    # Volume Volatility Dynamics
    # Volume Volatility Components
    data['Volume_Volatility_Ratio'] = (data['volume'] / data['volume'].shift(1)) - (data['volume'].shift(2) / data['volume'].shift(3))
    data['Volume_Volatility_Divergence'] = (data['volume'] / data['volume'].shift(2)) - (data['volume'].shift(1) / data['volume'].shift(3))
    data['Volume_Volatility_Momentum'] = data['Volume_Volatility_Ratio'] * data['Volume_Volatility_Divergence']
    
    # Price-Volatility Alignment
    data['Volatility_Direction_Consistency'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign((data['high'] - data['low']) - (data['high'].shift(1) - data['low'].shift(1)))
    data['Volatility_Magnitude_Divergence'] = ((data['high'] - data['low']) / data['close'].shift(1)) - ((data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1))
    data['Volatility_Divergence_Confirmation'] = data['Volatility_Direction_Consistency'] * data['Volatility_Magnitude_Divergence']
    
    # Fractal Volume Volatility
    data['Morning_Volatility_Fractal'] = data['volume'] * data['Opening_Volatility_Strength']
    data['Afternoon_Volatility_Fractal'] = data['volume'] * data['Closing_Volatility_Strength']
    data['Volume_Volatility_Fractal_Divergence'] = data['Morning_Volatility_Fractal'] - data['Afternoon_Volatility_Fractal']
    
    # Multi-Scale Volatility Integration
    data['Short_term_Volatility'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    # Medium-term Volatility (5-day rolling average)
    data['High_Low_Range'] = data['high'] - data['low']
    data['Medium_term_Volatility'] = data['High_Low_Range'].rolling(window=5).mean() / data['close'].shift(5)
    
    data['Volatility_Acceleration'] = data['Short_term_Volatility'] - data['Medium_term_Volatility']
    
    # Fractal Volatility Alignment
    data['Micro_Fractal_Volatility'] = data['Short_term_Volatility'] * np.sign(data['Volatility_Strength_Divergence'])
    data['Meso_Fractal_Volatility'] = data['Medium_term_Volatility'] * np.sign(data['Volatility_Persistence_Divergence'])
    data['Macro_Fractal_Volatility'] = data['Volatility_Acceleration'] * np.sign(data['Volatility_Momentum_Divergence'])
    
    # Gap and Volatility Patterns
    data['Gap_Volatility'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['Intraday_Volatility_Reversal'] = (data['high'] - data['low']) / (abs(data['open'] - data['close'].shift(1)) + 0.001)
    data['Gap_Volatility_Strength'] = data['Gap_Volatility'] * data['Intraday_Volatility_Reversal']
    data['Fractal_Gap_Volatility'] = data['Gap_Volatility_Strength'] * data['Volume_Volatility_Fractal_Divergence']
    
    # Range Volatility Efficiency
    data['Range_Volatility_Utilization'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['Volume_Volatility_Concentration'] = (data['volume'] / data['volume'].rolling(window=3).sum()) * (data['high'] - data['low']) / data['close'].shift(1)
    data['Volatility_Efficiency_Signal'] = data['Range_Volatility_Utilization'] * data['Volume_Volatility_Concentration']
    data['Fractal_Volatility_Efficiency'] = data['Volatility_Efficiency_Signal'] * data['Volatility_Strength_Divergence']
    
    # Volatility Pressure Analysis
    data['Upper_Volatility_Pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)) - ((data['high'].shift(1) - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    data['Lower_Volatility_Pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 0.001)) - ((data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    data['Volatility_Pressure_Momentum'] = data['Upper_Volatility_Pressure'] - data['Lower_Volatility_Pressure']
    data['Fractal_Volatility_Pressure'] = data['Volatility_Pressure_Momentum'] * data['Volatility_Momentum_Divergence']
    
    # Amount-Based Volatility Divergence
    data['Trade_Volatility_Intensity'] = (data['amount'] / data['volume']) * (data['high'] - data['low']) / data['close'].shift(1)
    data['Trade_Volatility_Divergence'] = (data['amount'] / data['amount'].shift(1)) - (data['volume'] / data['volume'].shift(1)) * ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    data['Large_Trade_Volatility'] = (data['amount'] / data['amount'].rolling(window=3).mean()) * (data['high'] - data['low']) / data['High_Low_Range'].rolling(window=3).mean()
    data['Fractal_Trade_Volatility'] = data['Trade_Volatility_Divergence'] * data['Large_Trade_Volatility'] * data['Volume_Volatility_Fractal_Divergence']
    
    # Volatility Quality Assessment
    # Volatility Pattern Quality
    data['Volatility_Consistency'] = 0
    for i in range(1, len(data)):
        if np.sign(data['Volatility_Strength_Divergence'].iloc[i]) == np.sign(data['Volatility_Strength_Divergence'].iloc[i-1]):
            data['Volatility_Consistency'].iloc[i] = data['Volatility_Consistency'].iloc[i-1] + 1
        else:
            data['Volatility_Consistency'].iloc[i] = 1
    
    # Calculate sign changes for Volatility_Strength_Divergence
    data['Volatility_Sign_Change'] = (np.sign(data['Volatility_Strength_Divergence']) != np.sign(data['Volatility_Strength_Divergence'].shift(1))).astype(int)
    data['Volatility_Sign_Change_Count'] = data['Volatility_Sign_Change'].rolling(window=len(data), min_periods=1).sum()
    data['Volatility_Stability_Ratio'] = data['Volatility_Consistency'] / (data['Volatility_Sign_Change_Count'] + 1)
    data['Volatility_Pattern_Quality'] = data['Volatility_Consistency'] * data['Volatility_Stability_Ratio']
    
    # Volume Volatility Quality
    data['Volume_Volatility_Consistency'] = 0
    for i in range(1, len(data)):
        if np.sign(data['Volume_Volatility_Fractal_Divergence'].iloc[i]) == np.sign(data['Volume_Volatility_Fractal_Divergence'].iloc[i-1]):
            data['Volume_Volatility_Consistency'].iloc[i] = data['Volume_Volatility_Consistency'].iloc[i-1] + 1
        else:
            data['Volume_Volatility_Consistency'].iloc[i] = 1
    
    # Calculate sign changes for Volume_Volatility_Fractal_Divergence
    data['Volume_Volatility_Sign_Change'] = (np.sign(data['Volume_Volatility_Fractal_Divergence']) != np.sign(data['Volume_Volatility_Fractal_Divergence'].shift(1))).astype(int)
    data['Volume_Volatility_Sign_Change_Count'] = data['Volume_Volatility_Sign_Change'].rolling(window=len(data), min_periods=1).sum()
    data['Volume_Volatility_Stability'] = data['Volume_Volatility_Consistency'] / (data['Volume_Volatility_Sign_Change_Count'] + 1)
    data['Volume_Volatility_Quality'] = data['Volume_Volatility_Consistency'] * data['Volume_Volatility_Stability']
    
    # Adaptive Volatility Divergence Synthesis
    data['Core_Volatility_Divergence'] = data['Volatility_Momentum_Divergence'] * data['Volume_Volatility_Divergence'] * data['Volatility_Divergence_Confirmation']
    data['Enhanced_Volatility_Divergence'] = data['Core_Volatility_Divergence'] * data['Fractal_Gap_Volatility'] * data['Fractal_Volatility_Efficiency']
    data['Pressure_Weighted_Volatility'] = data['Enhanced_Volatility_Divergence'] * data['Fractal_Volatility_Pressure'] * data['Fractal_Trade_Volatility']
    data['Quality_Enhanced_Volatility'] = data['Pressure_Weighted_Volatility'] * data['Volatility_Pattern_Quality'] * data['Volume_Volatility_Quality']
    data['Final_Volatility_Divergence_Alpha'] = data['Quality_Enhanced_Volatility'] * np.sign(data['Volatility_Acceleration'])
    
    return data['Final_Volatility_Divergence_Alpha']
