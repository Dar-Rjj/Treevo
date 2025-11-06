import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Microstructure Fracture Analysis
    # Price Micro-Fractures
    data['Micro_Gap_Intensity'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['Intraday_Fracture_Momentum'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * np.sign(data['close'] - data['close'].shift(1))
    
    # Fracture Persistence
    data['price_change_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['consecutive_same_sign'] = 0
    for i in range(1, len(data)):
        if data['price_change_sign'].iloc[i] == data['price_change_sign'].iloc[i-1]:
            data['consecutive_same_sign'].iloc[i] = data['consecutive_same_sign'].iloc[i-1] + 1
        else:
            data['consecutive_same_sign'].iloc[i] = 1
    data['Fracture_Persistence'] = data['consecutive_same_sign']
    
    # Volume Micro-Structure
    data['Volume_Fracture_Ratio'] = (data['volume'] / data['volume'].shift(1)) * ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)))
    data['Micro_Flow_Asymmetry'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) - ((data['high'] - data['close']) / (data['high'] - data['low']))
    data['Volume_Pressure_Divergence'] = (data['volume'] * (data['close'] - data['open'])) - (data['volume'].shift(1) * (data['close'].shift(1) - data['open'].shift(1)))
    
    # Microstructure Integration
    data['Fracture_Volume_Alignment'] = data['Micro_Gap_Intensity'] * data['Volume_Fracture_Ratio']
    data['Flow_Persistence_Coupling'] = data['Intraday_Fracture_Momentum'] * data['Fracture_Persistence']
    data['Micro_Pressure_Synthesis'] = data['Micro_Flow_Asymmetry'] * data['Volume_Pressure_Divergence']
    
    # Volatility-Resilient Momentum
    # Adaptive Momentum Measures
    data['Volatility_Adjusted_Momentum'] = (data['close'] - data['close'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2))
    data['Fracture_Enhanced_Acceleration'] = ((data['close'] - data['close'].shift(1)) / data['close'].shift(1)) * data['Fracture_Persistence']
    data['Momentum_Volatility_Ratio'] = ((data['close'] - data['close'].shift(3)) / (data['high'].shift(3) - data['low'].shift(3))) - ((data['close'].shift(3) - data['close'].shift(6)) / (data['high'].shift(6) - data['low'].shift(6)))
    
    # Volatility Regime Detection
    data['High_Volatility_Flag'] = (data['high'] - data['low']) > (data['high'].shift(5) - data['low'].shift(5)) * 1.2
    data['Low_Volatility_Flag'] = (data['high'] - data['low']) < (data['high'].shift(5) - data['low'].shift(5)) * 0.8
    data['Volatility_Transition'] = np.abs((data['high'] - data['low']) - (data['high'].shift(5) - data['low'].shift(5))) / (data['high'].shift(5) - data['low'].shift(5))
    
    # Resilience-Enhanced Momentum
    data['High_Volatility_Momentum'] = data['Volatility_Adjusted_Momentum'] * data['High_Volatility_Flag']
    data['Low_Volatility_Momentum'] = data['Fracture_Enhanced_Acceleration'] * data['Low_Volatility_Flag']
    data['Transition_Momentum'] = data['Momentum_Volatility_Ratio'] * data['Volatility_Transition']
    
    # Microstructure-Momentum Synchronization
    # Fracture-Momentum Alignment
    data['Micro_Fracture_Momentum'] = data['Fracture_Volume_Alignment'] * data['Volatility_Adjusted_Momentum']
    data['Flow_Momentum_Persistence'] = data['Flow_Persistence_Coupling'] * data['Fracture_Enhanced_Acceleration']
    data['Pressure_Momentum_Divergence'] = data['Micro_Pressure_Synthesis'] * data['Momentum_Volatility_Ratio']
    
    # Volatility-Regime Synchronization
    data['High_Volatility_Sync'] = data['Micro_Fracture_Momentum'] * data['High_Volatility_Momentum']
    data['Low_Volatility_Sync'] = data['Flow_Momentum_Persistence'] * data['Low_Volatility_Momentum']
    data['Transition_Sync'] = data['Pressure_Momentum_Divergence'] * data['Transition_Momentum']
    
    # Synchronized Micro-Momentum
    data['High_Regime_Micro_Momentum'] = data['High_Volatility_Sync'] * data['Volume_Fracture_Ratio']
    data['Low_Regime_Micro_Momentum'] = data['Low_Volatility_Sync'] * data['Micro_Flow_Asymmetry']
    data['Transition_Micro_Momentum'] = data['Transition_Sync'] * data['Volume_Pressure_Divergence']
    
    # Anchoring Signal Enhancement
    # Microstructure Anchors
    data['Volume_Anchor'] = (data['volume'] / (data['high'] - data['low'])) * data['Fracture_Persistence']
    data['Flow_Anchor'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * data['Micro_Gap_Intensity']
    data['Pressure_Anchor'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) * data['volume']
    
    # Momentum Anchoring
    data['Volume_Anchored_Momentum'] = data['Volatility_Adjusted_Momentum'] * data['Volume_Anchor']
    data['Flow_Anchored_Momentum'] = data['Fracture_Enhanced_Acceleration'] * data['Flow_Anchor']
    data['Pressure_Anchored_Momentum'] = data['Momentum_Volatility_Ratio'] * data['Pressure_Anchor']
    
    # Anchored Signal Integration
    data['High_Volatility_Anchor'] = data['Volume_Anchored_Momentum'] * data['High_Volatility_Flag']
    data['Low_Volatility_Anchor'] = data['Flow_Anchored_Momentum'] * data['Low_Volatility_Flag']
    data['Transition_Anchor'] = data['Pressure_Anchored_Momentum'] * data['Volatility_Transition']
    
    # Composite Factor Construction
    # Core Synchronization Layer
    data['Micro_Momentum_Core'] = data['High_Regime_Micro_Momentum'] + data['Low_Regime_Micro_Momentum'] + data['Transition_Micro_Momentum']
    data['Anchor_Enhanced_Core'] = data['Micro_Momentum_Core'] * (data['Volume_Anchor'] + data['Flow_Anchor'] + data['Pressure_Anchor'])
    data['Volatility_Resilient_Core'] = data['Anchor_Enhanced_Core'] * (1 - data['Volatility_Transition'])
    
    # Regime-Adaptive Refinement
    data['High_Volatility_Factor'] = data['Volatility_Resilient_Core'] * data['High_Volatility_Anchor']
    data['Low_Volatility_Factor'] = data['Volatility_Resilient_Core'] * data['Low_Volatility_Anchor']
    data['Transition_Factor'] = data['Volatility_Resilient_Core'] * data['Transition_Anchor']
    
    # Final Composite Integration
    data['Regime_Weighted_Composite'] = data['High_Volatility_Factor'] + data['Low_Volatility_Factor'] + data['Transition_Factor']
    data['Microstructure_Finalization'] = data['Regime_Weighted_Composite'] * data['Micro_Flow_Asymmetry']
    data['Momentum_Finalization'] = data['Microstructure_Finalization'] * data['Fracture_Enhanced_Acceleration']
    
    # Alpha Factor Output
    data['Volatility_Resilient_Alpha'] = data['Momentum_Finalization'] * data['Fracture_Persistence']
    data['Microstructure_Anchored_Alpha'] = data['Volatility_Resilient_Alpha'] * data['Volume_Anchor']
    data['Final_Alpha'] = data['Microstructure_Anchored_Alpha'] * data['Flow_Anchor']
    
    return data['Final_Alpha']
