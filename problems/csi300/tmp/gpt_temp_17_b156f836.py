import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Pressure Asymmetry Components
    # Intraday Pressure Asymmetry
    data['Morning_Pressure_Intensity'] = (data['open'] - data['low'])**2 / (data['high'] - data['open'] + 0.001)
    data['Afternoon_Pressure_Intensity'] = (data['high'] - data['close'])**2 / (data['close'] - data['low'] + 0.001)
    data['Intraday_Pressure_Asymmetry'] = data['Morning_Pressure_Intensity'] - data['Afternoon_Pressure_Intensity']
    
    # Multi-Scale Pressure Asymmetry
    data['Short_Term_Pressure_Asymmetry'] = data['Intraday_Pressure_Asymmetry'] - data['Intraday_Pressure_Asymmetry'].shift(1)
    data['Medium_Term_Pressure_Asymmetry'] = data['Intraday_Pressure_Asymmetry'] - data['Intraday_Pressure_Asymmetry'].shift(3)
    data['Long_Term_Pressure_Asymmetry'] = data['Intraday_Pressure_Asymmetry'] - data['Intraday_Pressure_Asymmetry'].shift(8)
    
    # Pressure Asymmetry Quality
    data['Pressure_Consistency'] = np.sign(data['Short_Term_Pressure_Asymmetry']) * np.sign(data['Medium_Term_Pressure_Asymmetry']) * np.sign(data['Long_Term_Pressure_Asymmetry'])
    data['Pressure_Magnitude'] = np.abs(data['Short_Term_Pressure_Asymmetry']) * np.abs(data['Medium_Term_Pressure_Asymmetry']) * np.abs(data['Long_Term_Pressure_Asymmetry'])
    
    # Pressure_Persistence calculation
    def calculate_pressure_persistence(series):
        if len(series) < 4:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        count = 0
        for i in range(1, min(4, len(series))):
            if np.sign(series.iloc[-i-1]) == current_sign:
                count += 1
        return count
    
    data['Pressure_Persistence'] = data['Pressure_Consistency'].rolling(window=4, min_periods=1).apply(calculate_pressure_persistence, raw=False)
    
    # Volume-Pressure Fractal Dynamics
    # Volume Pressure Components
    data['Volume_Pressure_Flow'] = (data['volume'] / data['volume'].shift(1)) * data['Intraday_Pressure_Asymmetry']
    data['Volume_Pressure_Gradient'] = ((data['high'] - data['low']) / (data['volume'] / data['volume'].shift(1) + 0.001)) * np.sign(data['Intraday_Pressure_Asymmetry'])
    data['Volume_Pressure_Transmission'] = (data['volume'] / data['volume'].shift(1)) * ((data['open'] - data['close'].shift(1))**2 / (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    
    # Pressure-Volume Divergence
    data['Flow_Pressure_Divergence'] = (np.sign(data['Volume_Pressure_Flow']) != np.sign(data['Intraday_Pressure_Asymmetry'])).astype(int)
    data['Gradient_Pressure_Divergence'] = (np.sign(data['Volume_Pressure_Gradient']) != np.sign(data['Intraday_Pressure_Asymmetry'])).astype(int)
    data['Transmission_Pressure_Divergence'] = (np.sign(data['Volume_Pressure_Transmission']) != np.sign(data['Intraday_Pressure_Asymmetry'])).astype(int)
    
    # Volume-Weighted Pressure
    data['Pressure_Divergence_Count'] = data['Flow_Pressure_Divergence'] + data['Gradient_Pressure_Divergence'] + data['Transmission_Pressure_Divergence']
    data['Volume_Pressure_Weight'] = data['Pressure_Divergence_Count'] * (data['volume'] / data['volume'].shift(1))
    data['Weighted_Pressure_Asymmetry'] = data['Volume_Pressure_Weight'] * data['Pressure_Consistency']
    
    # Amount-Efficiency Pressure Patterns
    # Trade Efficiency Pressure
    data['Opening_Trade_Pressure'] = (data['close'] - data['open']) * (data['amount'] / data['volume'])
    data['High_Trade_Pressure'] = (data['high'] - data['open']) * (data['amount'] / data['volume'])
    data['Low_Trade_Pressure'] = (data['open'] - data['low']) * (data['amount'] / data['volume'])
    
    # Efficiency Pressure Asymmetry
    data['Opening_High_Pressure_Asymmetry'] = data['Opening_Trade_Pressure'] - data['High_Trade_Pressure']
    data['Opening_Low_Pressure_Asymmetry'] = data['Opening_Trade_Pressure'] - data['Low_Trade_Pressure']
    data['High_Low_Pressure_Asymmetry'] = data['High_Trade_Pressure'] - data['Low_Trade_Pressure']
    
    # Multi-Scale Efficiency Pressure
    data['Short_Efficiency_Pressure'] = data['Opening_High_Pressure_Asymmetry'] - data['Opening_High_Pressure_Asymmetry'].shift(1)
    data['Medium_Efficiency_Pressure'] = data['Opening_Low_Pressure_Asymmetry'] - data['Opening_Low_Pressure_Asymmetry'].shift(3)
    data['Long_Efficiency_Pressure'] = data['High_Low_Pressure_Asymmetry'] - data['High_Low_Pressure_Asymmetry'].shift(8)
    
    # Fractal Pressure Transmission Framework
    # Pressure Persistence
    def calculate_persistence(series, window_size):
        if len(series) < window_size + 1:
            return np.nan
        current_sign = np.sign(series.iloc[-1])
        count = 0
        for i in range(1, min(window_size + 1, len(series))):
            if np.sign(series.iloc[-i-1]) == current_sign:
                count += 1
        return count
    
    data['Short_Pressure_Persistence'] = data['Short_Term_Pressure_Asymmetry'].rolling(window=3, min_periods=1).apply(lambda x: calculate_persistence(x, 2), raw=False)
    data['Medium_Pressure_Persistence'] = data['Medium_Term_Pressure_Asymmetry'].rolling(window=4, min_periods=1).apply(lambda x: calculate_persistence(x, 3), raw=False)
    data['Long_Pressure_Persistence'] = data['Long_Term_Pressure_Asymmetry'].rolling(window=6, min_periods=1).apply(lambda x: calculate_persistence(x, 5), raw=False)
    
    # Efficiency Pressure Persistence
    data['Short_Efficiency_Pressure_Persistence'] = data['Short_Efficiency_Pressure'].rolling(window=3, min_periods=1).apply(lambda x: calculate_persistence(x, 2), raw=False)
    data['Medium_Efficiency_Pressure_Persistence'] = data['Medium_Efficiency_Pressure'].rolling(window=4, min_periods=1).apply(lambda x: calculate_persistence(x, 3), raw=False)
    data['Long_Efficiency_Pressure_Persistence'] = data['Long_Efficiency_Pressure'].rolling(window=6, min_periods=1).apply(lambda x: calculate_persistence(x, 5), raw=False)
    
    # Combined Pressure Quality
    data['Pressure_Persistence_Score'] = data['Short_Pressure_Persistence'] * data['Medium_Pressure_Persistence'] * data['Long_Pressure_Persistence']
    data['Efficiency_Pressure_Persistence_Score'] = data['Short_Efficiency_Pressure_Persistence'] * data['Medium_Efficiency_Pressure_Persistence'] * data['Long_Efficiency_Pressure_Persistence']
    data['Fractal_Pressure_Quality'] = data['Pressure_Persistence_Score'] * data['Efficiency_Pressure_Persistence_Score']
    
    # Hierarchical Pressure Integration
    # Core Pressure Components
    data['Pressure_Core'] = data['Pressure_Consistency'] * data['Pressure_Magnitude'] * data['Pressure_Persistence']
    data['Volume_Pressure_Core'] = data['Weighted_Pressure_Asymmetry'] * data['Volume_Pressure_Weight']
    data['Efficiency_Pressure_Core'] = data['High_Low_Pressure_Asymmetry'] * data['Opening_Trade_Pressure']
    
    # Multi-Scale Pressure Enhancement
    data['Persistence_Enhanced_Pressure'] = data['Pressure_Core'] * data['Fractal_Pressure_Quality']
    data['Volume_Enhanced_Pressure'] = data['Volume_Pressure_Core'] * data['Volume_Pressure_Flow']
    data['Efficiency_Enhanced_Pressure'] = data['Efficiency_Pressure_Core'] * data['Opening_High_Pressure_Asymmetry']
    
    # Pressure Alignment Dynamics
    data['Pressure_Volume_Alignment'] = np.sign(data['Persistence_Enhanced_Pressure']) * np.sign(data['Volume_Enhanced_Pressure'])
    data['Pressure_Efficiency_Alignment'] = np.sign(data['Persistence_Enhanced_Pressure']) * np.sign(data['Efficiency_Enhanced_Pressure'])
    data['Multi_Scale_Pressure_Coherence'] = data['Pressure_Volume_Alignment'] * data['Pressure_Efficiency_Alignment']
    
    # Final Alpha Construction
    # Base Alpha Pressure Components
    data['Pressure_Volume_Base'] = data['Persistence_Enhanced_Pressure'] * data['Volume_Enhanced_Pressure']
    data['Pressure_Efficiency_Base'] = data['Efficiency_Enhanced_Pressure'] * data['Fractal_Pressure_Quality']
    data['Core_Pressure_Base'] = data['Pressure_Core'] * data['Volume_Pressure_Core']
    
    # Hierarchical Pressure Integration
    data['Core_Pressure_Alpha'] = data['Pressure_Volume_Base'] * data['Pressure_Efficiency_Base'] * data['Core_Pressure_Base']
    data['Multi_Scale_Pressure_Enhancement'] = data['Core_Pressure_Alpha'] * data['Multi_Scale_Pressure_Coherence']
    data['Fractal_Pressure_Acceleration'] = data['Multi_Scale_Pressure_Enhancement'] * (data['close'] - data['open'])
    
    # Final Alpha Factor
    data['Multi_Scale_Fractal_Pressure_Volume_Asymmetry_Alpha'] = data['Fractal_Pressure_Acceleration'] * np.sign(
        data['Short_Term_Pressure_Asymmetry'] + data['Medium_Term_Pressure_Asymmetry'] + data['Long_Term_Pressure_Asymmetry']
    )
    
    return data['Multi_Scale_Fractal_Pressure_Volume_Asymmetry_Alpha']
