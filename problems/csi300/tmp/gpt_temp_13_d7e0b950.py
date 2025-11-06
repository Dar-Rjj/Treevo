import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Price Divergence Structure
    # Opening Fractal Divergence
    data['Opening_Fractal_Strength'] = (data['open'] - data['low']) / (data['high'] - data['low'] + 0.001)
    
    # Calculate consecutive days where Opening_Fractal_Strength > 0.5
    data['Opening_Fractal_Persistence'] = 0
    for i in range(1, len(data)):
        if data['Opening_Fractal_Strength'].iloc[i] > 0.5:
            if data['Opening_Fractal_Strength'].iloc[i-1] > 0.5:
                data['Opening_Fractal_Persistence'].iloc[i] = data['Opening_Fractal_Persistence'].iloc[i-1] + 1
            else:
                data['Opening_Fractal_Persistence'].iloc[i] = 1
        else:
            data['Opening_Fractal_Persistence'].iloc[i] = 0
    
    data['Opening_Fractal_Momentum'] = data['Opening_Fractal_Strength'] * data['Opening_Fractal_Persistence']
    
    # Closing Fractal Divergence
    data['Closing_Fractal_Strength'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 0.001)
    
    # Calculate consecutive days where Closing_Fractal_Strength > 0.5
    data['Closing_Fractal_Persistence'] = 0
    for i in range(1, len(data)):
        if data['Closing_Fractal_Strength'].iloc[i] > 0.5:
            if data['Closing_Fractal_Strength'].iloc[i-1] > 0.5:
                data['Closing_Fractal_Persistence'].iloc[i] = data['Closing_Fractal_Persistence'].iloc[i-1] + 1
            else:
                data['Closing_Fractal_Persistence'].iloc[i] = 1
        else:
            data['Closing_Fractal_Persistence'].iloc[i] = 0
    
    data['Closing_Fractal_Momentum'] = data['Closing_Fractal_Strength'] * data['Closing_Fractal_Persistence']
    
    # Fractal Pattern Divergence
    data['Fractal_Strength_Divergence'] = data['Opening_Fractal_Strength'] - data['Closing_Fractal_Strength']
    data['Fractal_Persistence_Divergence'] = data['Opening_Fractal_Persistence'] - data['Closing_Fractal_Persistence']
    data['Fractal_Momentum_Divergence'] = data['Opening_Fractal_Momentum'] - data['Closing_Fractal_Momentum']
    
    # Volume Divergence Dynamics
    # Volume Momentum Components
    data['Volume_Acceleration'] = (data['volume'] / data['volume'].shift(1) - 
                                  data['volume'].shift(1) / data['volume'].shift(2))
    
    data['Volume_Divergence_Ratio'] = (data['volume'] / data['volume'].shift(1).rolling(window=3, min_periods=1).mean() - 1)
    
    data['Volume_Volatility'] = (data['volume'] / data['volume'].shift(3) - 
                                data['volume'].shift(1) / data['volume'].shift(4))
    
    # Price-Volume Alignment
    data['Direction_Consistency'] = (np.sign(data['close'] - data['close'].shift(1)) * 
                                    np.sign(data['volume'] - data['volume'].shift(1)))
    
    data['Magnitude_Divergence'] = (abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1) - 
                                   abs(data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1))
    
    data['Divergence_Confirmation'] = data['Direction_Consistency'] * data['Magnitude_Divergence']
    
    # Fractal Volume Microstructure
    data['Morning_Volume_Fractal'] = data['volume'] * data['Opening_Fractal_Strength']
    data['Afternoon_Volume_Fractal'] = data['volume'] * data['Closing_Fractal_Strength']
    data['Volume_Fractal_Divergence'] = data['Morning_Volume_Fractal'] - data['Afternoon_Volume_Fractal']
    
    # Multi-Scale Momentum Integration
    data['Short_term_Momentum'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['Medium_term_Momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['Acceleration_Momentum'] = data['Short_term_Momentum'] - data['Medium_term_Momentum']
    
    # Fractal Momentum Alignment
    data['Micro_Fractal_Momentum'] = data['Short_term_Momentum'] * np.sign(data['Fractal_Strength_Divergence'])
    data['Meso_Fractal_Momentum'] = data['Medium_term_Momentum'] * np.sign(data['Fractal_Persistence_Divergence'])
    data['Macro_Fractal_Momentum'] = data['Acceleration_Momentum'] * np.sign(data['Fractal_Momentum_Divergence'])
    
    # Gap and Reversal Patterns
    data['Gap_Size'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['Intraday_Reversal'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 0.001)
    data['Gap_Reversal_Strength'] = data['Gap_Size'] * data['Intraday_Reversal']
    data['Fractal_Gap_Momentum'] = data['Gap_Reversal_Strength'] * data['Volume_Fractal_Divergence']
    
    # Range Efficiency Divergence
    data['Range_Utilization'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    data['Volume_Concentration'] = data['volume'] / data['volume'].shift(1).rolling(window=2, min_periods=1).sum()
    data['Efficiency_Signal'] = data['Range_Utilization'] * data['Volume_Concentration']
    data['Fractal_Efficiency_Divergence'] = data['Efficiency_Signal'] * data['Fractal_Strength_Divergence']
    
    # Pressure Divergence Analysis
    data['Upper_Pressure_Divergence'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 0.001) - 
                                        (data['high'].shift(1) - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    
    data['Lower_Pressure_Divergence'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 0.001) - 
                                        (data['close'].shift(1) - data['low'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    
    data['Pressure_Momentum'] = data['Upper_Pressure_Divergence'] - data['Lower_Pressure_Divergence']
    data['Fractal_Pressure_Alignment'] = data['Pressure_Momentum'] * data['Fractal_Momentum_Divergence']
    
    # Amount-Based Fractal Divergence
    data['Trade_Intensity'] = data['amount'] / (data['volume'] + 0.001)
    data['Trade_Size_Divergence'] = (data['amount'] / data['amount'].shift(1) - 
                                    data['volume'] / data['volume'].shift(1))
    
    data['Large_Trade_Momentum'] = data['amount'] / data['amount'].shift(1).rolling(window=3, min_periods=1).mean()
    data['Fractal_Trade_Divergence'] = (data['Trade_Size_Divergence'] * data['Large_Trade_Momentum'] * 
                                       data['Volume_Fractal_Divergence'])
    
    # Fractal Quality Assessment
    # Fractal Pattern Quality
    data['Fractal_Consistency'] = 0
    for i in range(1, len(data)):
        if np.sign(data['Fractal_Strength_Divergence'].iloc[i]) == np.sign(data['Fractal_Strength_Divergence'].iloc[i-1]):
            data['Fractal_Consistency'].iloc[i] = data['Fractal_Consistency'].iloc[i-1] + 1
        else:
            data['Fractal_Consistency'].iloc[i] = 1
    
    # Calculate sign changes for Fractal_Strength_Divergence
    data['Fractal_Sign_Change'] = (np.sign(data['Fractal_Strength_Divergence']) != 
                                  np.sign(data['Fractal_Strength_Divergence'].shift(1))).astype(int)
    data['Fractal_Sign_Change_Count'] = data['Fractal_Sign_Change'].rolling(window=len(data), min_periods=1).sum()
    
    data['Fractal_Stability_Ratio'] = data['Fractal_Consistency'] / (data['Fractal_Sign_Change_Count'] + 1)
    data['Fractal_Pattern_Quality'] = data['Fractal_Consistency'] * data['Fractal_Stability_Ratio']
    
    # Volume Fractal Quality
    data['Volume_Fractal_Consistency'] = 0
    for i in range(1, len(data)):
        if np.sign(data['Volume_Fractal_Divergence'].iloc[i]) == np.sign(data['Volume_Fractal_Divergence'].iloc[i-1]):
            data['Volume_Fractal_Consistency'].iloc[i] = data['Volume_Fractal_Consistency'].iloc[i-1] + 1
        else:
            data['Volume_Fractal_Consistency'].iloc[i] = 1
    
    # Calculate sign changes for Volume_Fractal_Divergence
    data['Volume_Fractal_Sign_Change'] = (np.sign(data['Volume_Fractal_Divergence']) != 
                                         np.sign(data['Volume_Fractal_Divergence'].shift(1))).astype(int)
    data['Volume_Fractal_Sign_Change_Count'] = data['Volume_Fractal_Sign_Change'].rolling(window=len(data), min_periods=1).sum()
    
    data['Volume_Fractal_Stability'] = data['Volume_Fractal_Consistency'] / (data['Volume_Fractal_Sign_Change_Count'] + 1)
    data['Volume_Fractal_Quality'] = data['Volume_Fractal_Consistency'] * data['Volume_Fractal_Stability']
    
    # Adaptive Fractal Divergence Synthesis
    data['Core_Fractal_Divergence'] = (data['Fractal_Pressure_Alignment'] * data['Volume_Divergence_Ratio'] * 
                                      data['Divergence_Confirmation'])
    
    data['Enhanced_Fractal_Divergence'] = (data['Core_Fractal_Divergence'] * data['Fractal_Gap_Momentum'] * 
                                          data['Fractal_Efficiency_Divergence'])
    
    data['Pressure_Weighted_Divergence'] = (data['Enhanced_Fractal_Divergence'] * data['Fractal_Pressure_Alignment'] * 
                                           data['Fractal_Trade_Divergence'])
    
    data['Quality_Enhanced_Divergence'] = (data['Pressure_Weighted_Divergence'] * data['Fractal_Pattern_Quality'] * 
                                          data['Volume_Fractal_Quality'])
    
    data['Final_Fractal_Divergence_Alpha'] = data['Quality_Enhanced_Divergence'] * np.sign(data['Acceleration_Momentum'])
    
    # Return the final alpha factor
    return data['Final_Fractal_Divergence_Alpha']
