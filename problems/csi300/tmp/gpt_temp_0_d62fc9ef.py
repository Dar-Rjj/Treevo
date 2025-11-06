import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Asymmetric Volatility Fracture Detection
    # Range Acceleration Asymmetry
    data['Upper_Acceleration'] = (data['high'] - data['high'].shift(1)) / (data['high'].shift(1) - data['high'].shift(2))
    data['Lower_Acceleration'] = (data['low'] - data['low'].shift(1)) / (data['low'].shift(1) - data['low'].shift(2))
    data['Fracture_Signal'] = ((data['Upper_Acceleration'].abs() > 2.0) | (data['Lower_Acceleration'].abs() > 2.0)).astype(int)
    
    # Volatility Asymmetry Components
    data['max_open_close'] = data[['open', 'close']].max(axis=1)
    data['min_open_close'] = data[['open', 'close']].min(axis=1)
    
    data['Upside_Volatility_Fracture'] = (data['high'] - data['max_open_close']) * data['volume'] / (data['high'] - data['max_open_close'] + 1e-8)
    data['Downside_Volatility_Fracture'] = (data['min_open_close'] - data['low']) * data['volume'] / (data['min_open_close'] - data['low'] + 1e-8)
    data['Volatility_Asymmetry_Ratio'] = data['Upside_Volatility_Fracture'] / (data['Downside_Volatility_Fracture'] + 1e-8)
    
    # Fracture Confirmation
    data['Momentum_Slope_Change'] = (data['close'] - data['close'].shift(1)) - (data['close'].shift(1) - data['close'].shift(2))
    data['Range_Asymmetry'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) - 0.5
    data['Cross_Fracture'] = ((data['Range_Asymmetry'].abs() > 0.3) & 
                             ((data['Momentum_Slope_Change'] * data['Volatility_Asymmetry_Ratio']).abs() > 
                              data['Momentum_Slope_Change'].abs())).astype(int)
    
    # Asymmetric Liquidity Alignment
    # Volume Flow Asymmetry
    data['Upper_Flow'] = data['amount'] * (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['Lower_Flow'] = data['amount'] * (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['Volume_Fracture'] = data['Upper_Flow'] - data['Lower_Flow']
    
    # Volume Compression Dynamics
    data['Volume_Drying'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['Volume_Surge'] = data['volume'] / ((data['volume'].shift(1) + data['volume'].shift(2) + 
                                            data['volume'].shift(3) + data['volume'].shift(4) + 
                                            data['volume'].shift(5)) / 5 + 1e-8)
    data['Volume_Persistence'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * \
                                (data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8))
    
    # Price-Volume Alignment
    data['Directional_Imbalance'] = (np.maximum(0, data['close'] - data['close'].shift(1)) - 
                                   np.maximum(0, data['close'].shift(1) - data['close'])) / \
                                   (np.maximum(0, data['close'] - data['close'].shift(1)) + 
                                    np.maximum(0, data['close'].shift(1) - data['close']) + 1e-8)
    data['Volume_Price_Alignment'] = np.sign(data['volume'] - data['volume'].shift(5)) * np.sign(data['close'] - data['close'].shift(3))
    data['Microstructure_Flow'] = (data['close'] - data['close'].shift(1)) * data['volume'] / (data['amount'] + 1e-8)
    
    # Movement Efficiency calculation
    data['Movement_Efficiency'] = (data['close'] - data['close'].shift(1)).abs() / (data['high'] - data['low'] + 1e-8)
    
    # Multi-Timeframe Fracture-Alignment Integration
    # Volatility-Liquidity Coupling
    data['Volatility_Volume_Alignment'] = data['Volatility_Asymmetry_Ratio'] * data['Volume_Price_Alignment']
    data['Fracture_Microstructure_Confirmation'] = data['Cross_Fracture'] * data['Microstructure_Flow']
    data['Range_Efficiency_Integration'] = data['Movement_Efficiency'] * (data['Upside_Volatility_Fracture'] + data['Downside_Volatility_Fracture'])
    
    # Asymmetric Momentum Integration
    bullish_pressure = sum([np.maximum(0, data['close'].shift(i) - data['close'].shift(i+1)) for i in range(5)])
    bearish_pressure = sum([np.maximum(0, data['close'].shift(i+1) - data['close'].shift(i)) for i in range(5)])
    data['Momentum_Imbalance'] = (bullish_pressure - bearish_pressure) / (bullish_pressure + bearish_pressure + 1e-8)
    
    # Multi-Scale Alignment
    data['Short_Term_Alignment'] = data['Movement_Efficiency'] * data['Directional_Imbalance'] * data['Volume_Drying']
    data['Medium_Term_Alignment'] = data['Range_Asymmetry'] * data['Volume_Price_Alignment'] * data['Volume_Persistence']
    data['Timeframe_Convergence'] = np.sign(data['Short_Term_Alignment']) * np.sign(data['Medium_Term_Alignment'])
    
    # Asymmetric Cross-Conformation
    # Fracture-Momentum Alignment
    data['Fracture_Direction'] = np.sign(data['Upper_Acceleration'] - data['Lower_Acceleration'])
    data['Momentum_Direction'] = np.sign(data['Momentum_Imbalance'])
    data['Alignment_Signal'] = data['Fracture_Direction'] * data['Momentum_Direction']
    
    # Volume-Efficiency Convergence
    data['Volume_Asymmetry_Ratio'] = (data['volume'] / (data['volume'].shift(3) + 1e-8)) / \
                                    (np.abs(data['volume'] / (data['volume'].shift(5) + 1e-8)) + 0.001)
    data['Efficiency_Convergence'] = np.sign(data['Volume_Asymmetry_Ratio']) * np.sign(data['Movement_Efficiency'])
    data['Convergence_Signal'] = data['Efficiency_Convergence'] * data['Movement_Efficiency']
    
    # Fracture Intensity Assessment
    data['Immediate_Fracture'] = data['Cross_Fracture'] * data['Volatility_Volume_Alignment']
    data['Sustained_Fracture'] = data['Fracture_Microstructure_Confirmation'] * data['Timeframe_Convergence']
    data['Fracture_Intensity'] = data['Immediate_Fracture'] * data['Sustained_Fracture']
    
    # Adaptive Alpha Construction
    # Core Fracture Components
    data['Volatility_Fracture_Core'] = data['Momentum_Slope_Change'] * data['Volatility_Asymmetry_Ratio'] * data['Cross_Fracture']
    data['Liquidity_Alignment_Core'] = data['Volume_Price_Alignment'] * data['Volume_Fracture']
    data['Multi_Scale_Core'] = data['Timeframe_Convergence'] * data['Fracture_Intensity']
    
    # Asymmetric Enhancement
    data['Integrated_Fracture_Signal'] = data['Volatility_Fracture_Core'] * data['Liquidity_Alignment_Core']
    data['Strength_Validation'] = data['Fracture_Intensity'] * data['Alignment_Signal']
    data['Enhanced_Factor'] = data['Strength_Validation'] * data['Multi_Scale_Core']
    
    # Regime-Adaptive Output
    data['High_Fracture_Alpha'] = data['Integrated_Fracture_Signal'] * data['Enhanced_Factor'] * data['Fracture_Signal']
    data['Low_Fracture_Alpha'] = data['Integrated_Fracture_Signal'] * data['Enhanced_Factor'] * (1 - data['Fracture_Signal'])
    data['Base_Alpha'] = data['High_Fracture_Alpha'] + data['Low_Fracture_Alpha']
    
    # Quality-Adjusted Final Alpha
    # Signal Persistence
    data['Alignment_Persistence'] = data['Alignment_Signal'].rolling(window=3, min_periods=1).apply(lambda x: (x > 0).sum())
    data['Convergence_Persistence'] = data['Convergence_Signal'].rolling(window=3, min_periods=1).apply(lambda x: (x > 0).sum())
    data['Signal_Quality'] = data['Alignment_Persistence'] * data['Convergence_Persistence']
    
    # Final Asymmetric Alpha
    data['Quality_Factor'] = data['Base_Alpha'] * data['Signal_Quality']
    data['Volatility_Adjustment'] = (data['Upper_Acceleration'] - data['Lower_Acceleration']) * data['Volatility_Asymmetry_Ratio']
    data['Final_Alpha'] = data['Quality_Factor'] * (1 + data['Volatility_Adjustment']) * np.sign(data['Timeframe_Convergence'])
    
    return data['Final_Alpha']
