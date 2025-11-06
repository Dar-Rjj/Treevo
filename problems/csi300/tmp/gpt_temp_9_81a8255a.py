import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Fracture Alpha Framework v2
    Generates a composite alpha factor based on momentum stress, regime detection, 
    efficiency metrics, and multi-scale integration.
    """
    data = df.copy()
    
    # Core Asymmetry Detection
    data['Intraday_Asymmetry'] = (data['high'] - data['close']) - (data['close'] - data['low'])
    data['Volume_Change'] = data['volume'] / data['volume'].shift(1)
    data['Range_Change'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Momentum Stress Signals
    data['Momentum_Stress_Gap'] = (
        np.abs(data['Intraday_Asymmetry'] * np.sign(data['Volume_Change'] - 1)) - 
        np.abs(data['Intraday_Asymmetry'].shift(1) * np.sign(data['Volume_Change'].shift(1) - 1))
    )
    
    data['Fractal_Momentum_Intensity'] = (
        (data['high'] - data['low']) * 
        data['Momentum_Stress_Gap'] * 
        data['Intraday_Asymmetry']
    )
    
    # Volatility Persistence (3-day window)
    vol_persistence = []
    for i in range(len(data)):
        if i < 2:
            vol_persistence.append(np.nan)
            continue
        
        count = 0
        for j in range(i-2, i+1):
            if j < 1:
                continue
            current_sign = np.sign(data['Intraday_Asymmetry'].iloc[j] * np.sign(data['Volume_Change'].iloc[j] - 1))
            prev_sign = np.sign(data['Intraday_Asymmetry'].iloc[j-1] * np.sign(data['Volume_Change'].iloc[j-1] - 1))
            if current_sign == prev_sign:
                count += 1
        vol_persistence.append(count)
    data['Volatility_Persistence'] = vol_persistence
    
    # Regime Detection
    data['Expansion'] = (
        (data['Range_Change'] > 1.3) & 
        (data['Intraday_Asymmetry'] > 1.1 * data['Intraday_Asymmetry'].shift(1))
    ).astype(int)
    
    data['Contraction'] = (
        (data['Range_Change'] < 0.7) & 
        (data['Intraday_Asymmetry'] < 0.9 * data['Intraday_Asymmetry'].shift(1))
    ).astype(int)
    
    # Regime Persistence (3-day window)
    regime_persistence = []
    for i in range(len(data)):
        if i < 2:
            regime_persistence.append(np.nan)
            continue
        
        count = 0
        for j in range(i-2, i+1):
            if j < 1:
                continue
            if (data['Range_Change'].iloc[j] > 1) and (data['Intraday_Asymmetry'].iloc[j] > data['Intraday_Asymmetry'].iloc[j-1]):
                count += 1
        regime_persistence.append(count)
    data['Regime_Persistence'] = regime_persistence
    
    # Efficiency Metrics
    data['Range_Efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['Volume_Efficiency'] = (data['high'] - data['low']) / data['volume']
    
    # Efficiency Persistence (3-day window)
    efficiency_persistence = []
    for i in range(len(data)):
        if i < 2:
            efficiency_persistence.append(np.nan)
            continue
        
        count = 0
        for j in range(i-2, i+1):
            if j < 1:
                continue
            if (data['Range_Efficiency'].iloc[j] > data['Range_Efficiency'].iloc[j-1]) and (data['Intraday_Asymmetry'].iloc[j] > data['Intraday_Asymmetry'].iloc[j-1]):
                count += 1
        efficiency_persistence.append(count)
    data['Efficiency_Persistence'] = efficiency_persistence
    
    # Multi-Scale Integration
    data['Short_Term'] = (
        data['Momentum_Stress_Gap'] * 
        np.sign(data['Volume_Change'] - 1) * 
        np.sign(data['Intraday_Asymmetry']) * 
        np.sign(data['close'] - data['open']) * 
        data['Volume_Change']
    )
    
    data['Medium_Term'] = (
        (data['close'] - data['close'].shift(5)) * 
        (data['Intraday_Asymmetry'] - data['Intraday_Asymmetry'].shift(3)) * 
        (data['volume'] / data['volume'].shift(5))
    )
    
    data['Volume_Spike'] = data['volume'] / (
        (data['volume'].shift(4) + data['volume'].shift(3) + 
         data['volume'].shift(2) + data['volume'].shift(1)) / 4
    )
    
    # Signal Validation
    data['Price_Volume_Alignment'] = (
        np.sign(data['Intraday_Asymmetry']) * 
        np.sign(data['Volume_Change'] - 1)
    )
    
    data['Range_Consistency'] = (
        np.sign(data['Range_Change'] - 1) * 
        np.sign(data['Intraday_Asymmetry'])
    )
    
    data['Multi_Scale_Alignment'] = (
        np.sign(data['Short_Term']) * 
        np.sign(data['Medium_Term']) * 
        np.sign(data['Volume_Change'])
    )
    
    # Novel Asymmetry Features
    data['Gap_Asymmetry'] = (
        (data['open'] - data['close'].shift(1)) * 
        np.sign(data['Intraday_Asymmetry'])
    )
    
    data['Volume_Asymmetry_Divergence'] = (
        data['Volume_Change'] * 
        (data['Intraday_Asymmetry'] - data['Intraday_Asymmetry'].shift(1))
    )
    
    data['Range_Volume_Efficiency'] = (
        data['Range_Efficiency'] * 
        data['Volume_Efficiency'] * 
        np.sign(data['Intraday_Asymmetry'])
    )
    
    # Enhanced Momentum Signals
    # Stress Momentum Persistence (4-day window)
    stress_persistence = []
    for i in range(len(data)):
        if i < 3:
            stress_persistence.append(np.nan)
            continue
        
        count = 0
        for j in range(i-3, i+1):
            if j < 1:
                continue
            if data['Momentum_Stress_Gap'].iloc[j] > data['Momentum_Stress_Gap'].iloc[j-1]:
                count += 1
        stress_persistence.append(count)
    data['Stress_Momentum_Persistence'] = stress_persistence
    
    data['Fractal_Volume_Intensity'] = (
        data['Fractal_Momentum_Intensity'] * 
        data['Volume_Spike'] * 
        np.sign(data['Volume_Change'] - 1)
    )
    
    data['Multi_Timeframe_Alignment'] = (
        np.sign(data['Short_Term']) * 
        np.sign(data['Medium_Term']) * 
        np.sign(data['Volume_Spike'] - 1)
    )
    
    # Alpha Synthesis
    data['Price_Core'] = (
        data['Short_Term'] * 
        data['Price_Volume_Alignment'] * 
        data['Volume_Change']
    )
    
    data['Volatility_Core'] = (
        data['Fractal_Momentum_Intensity'] * 
        np.sign(data['Volume_Change'] - 1) * 
        np.sign(data['Intraday_Asymmetry']) * 
        np.sign(data['close'] - data['open'])
    )
    
    data['Volume_Core'] = (
        data['Volume_Spike'] * 
        data['Price_Volume_Alignment'] * 
        data['Intraday_Asymmetry'] / data['volume'] * 
        data['Range_Efficiency']
    )
    
    data['Efficiency_Core'] = (
        data['Range_Efficiency'] * 
        data['Efficiency_Persistence'] * 
        data['Range_Efficiency']
    )
    
    data['Novel_Core'] = (
        data['Gap_Asymmetry'] * 
        data['Volume_Asymmetry_Divergence'] * 
        data['Range_Volume_Efficiency']
    )
    
    # Composite Alpha
    data['Composite_Alpha'] = (
        (data['Price_Core'] + data['Volatility_Core'] + data['Volume_Core'] + 
         data['Efficiency_Core'] + data['Novel_Core']) * 
        data['Multi_Scale_Alignment'] * 
        data['Multi_Timeframe_Alignment']
    )
    
    return data['Composite_Alpha']
