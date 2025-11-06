import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add small epsilon to avoid division by zero
    eps = 0.001
    
    # Fractal Momentum Components
    # Micro-Scale Momentum
    data['Opening_Momentum_Strength'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + eps)) * (data['volume'] / data['volume'].shift(1))
    data['High_Momentum_Pressure'] = ((data['high'] - data['close'].shift(1)) / (data['close'].shift(1) - data['low'].shift(1) + eps)) * np.sign(data['close'] - data['open'])
    data['Low_Momentum_Support'] = ((data['close'].shift(1) - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + eps)) * np.sign(data['open'] - data['close'])
    
    # Meso-Scale Momentum (5-day window)
    data['Range_Momentum'] = (data['close'] - data['close'].shift(5)) / (data['high'].rolling(6).max() - data['low'].rolling(6).min() + eps)
    data['High_Pressure_Meso'] = (data['high'].rolling(6).max() - data['open']) / (data['high'].rolling(6).max() - data['low'].rolling(6).min() + eps)
    data['Low_Support_Meso'] = (data['open'] - data['low'].rolling(6).min()) / (data['high'].rolling(6).max() - data['low'].rolling(6).min() + eps)
    
    # Macro-Scale Momentum (13-day window)
    data['Extended_Range_Momentum'] = (data['close'] - data['close'].shift(13)) / (data['high'].rolling(14).max() - data['low'].rolling(14).min() + eps)
    data['High_Pressure_Macro'] = (data['high'].rolling(14).max() - data['open']) / (data['high'].rolling(14).max() - data['low'].rolling(14).min() + eps)
    data['Low_Support_Macro'] = (data['open'] - data['low'].rolling(14).min()) / (data['high'].rolling(14).max() - data['low'].rolling(14).min() + eps)
    
    # Fractal Flow Asymmetry
    data['Micro_Flow_Asymmetry'] = ((data['high'] - data['open']) / (data['high'] - data['low'] + eps)) - ((data['close'] - data['low']) / (data['high'] - data['low'] + eps))
    data['Meso_Flow_Asymmetry'] = ((data['high'].rolling(6).max() - data['open']) / (data['high'].rolling(6).max() - data['low'].rolling(6).min() + eps)) - ((data['open'] - data['low'].rolling(6).min()) / (data['high'].rolling(6).max() - data['low'].rolling(6).min() + eps))
    data['Macro_Flow_Asymmetry'] = ((data['high'].rolling(14).max() - data['open']) / (data['high'].rolling(14).max() - data['low'].rolling(14).min() + eps)) - ((data['open'] - data['low'].rolling(14).min()) / (data['high'].rolling(14).max() - data['low'].rolling(14).min() + eps))
    data['Fractal_Flow_Cascade'] = data['Micro_Flow_Asymmetry'] * data['Meso_Flow_Asymmetry'] * data['Macro_Flow_Asymmetry']
    
    # Volume-Momentum Regime Integration
    data['Morning_Volume_Flow'] = (data['high'] - data['open']) * data['volume'] / (data['high'] - data['low'] + eps)
    data['Afternoon_Volume_Flow'] = (data['close'] - data['low']) * data['volume'] / (data['high'] - data['low'] + eps)
    data['Volume_Flow_Asymmetry'] = data['Morning_Volume_Flow'] - data['Afternoon_Volume_Flow']
    
    data['Micro_Momentum_Volume'] = data['Opening_Momentum_Strength'] * data['Volume_Flow_Asymmetry']
    data['Meso_Momentum_Volume'] = data['Range_Momentum'] * data['Volume_Flow_Asymmetry']
    data['Macro_Momentum_Volume'] = data['Extended_Range_Momentum'] * data['Volume_Flow_Asymmetry']
    
    data['Momentum_Consistency'] = np.sign(data['Opening_Momentum_Strength']) * np.sign(data['Range_Momentum']) * np.sign(data['Extended_Range_Momentum'])
    data['Flow_Asymmetry_Consistency'] = np.sign(data['Micro_Flow_Asymmetry']) * np.sign(data['Meso_Flow_Asymmetry']) * np.sign(data['Macro_Flow_Asymmetry'])
    data['Fractal_Regime_Coherence'] = data['Momentum_Consistency'] * data['Flow_Asymmetry_Consistency']
    
    # Amount-Velocity Fractal Patterns
    data['Opening_Trade_Velocity'] = (data['close'] - data['open']) * data['amount'] / (data['volume'] + eps)
    data['High_Trade_Velocity'] = (data['high'] - data['open']) * data['amount'] / (data['volume'] + eps)
    data['Low_Trade_Velocity'] = (data['open'] - data['low']) * data['amount'] / (data['volume'] + eps)
    
    data['Micro_Velocity_Asymmetry'] = data['Opening_Trade_Velocity'] - data['High_Trade_Velocity']
    data['Meso_Velocity_Asymmetry'] = data['Opening_Trade_Velocity'] - data['Low_Trade_Velocity']
    data['Macro_Velocity_Asymmetry'] = data['High_Trade_Velocity'] - data['Low_Trade_Velocity']
    
    data['Micro_Velocity_Momentum'] = data['Micro_Velocity_Asymmetry'] * data['Opening_Momentum_Strength']
    data['Meso_Velocity_Momentum'] = data['Meso_Velocity_Asymmetry'] * data['Range_Momentum']
    data['Macro_Velocity_Momentum'] = data['Macro_Velocity_Asymmetry'] * data['Extended_Range_Momentum']
    
    # Fractal Regime Persistence
    # Calculate persistence counts using rolling windows
    for window in [2, 5, 13]:
        for component in ['Opening_Momentum_Strength', 'Range_Momentum', 'Extended_Range_Momentum', 
                         'Micro_Flow_Asymmetry', 'Meso_Flow_Asymmetry', 'Macro_Flow_Asymmetry']:
            if window == 2 and component in ['Opening_Momentum_Strength', 'Micro_Flow_Asymmetry']:
                persistence_col = f"{component.split('_')[0]}_{component.split('_')[1]}_Persistence"
                data[persistence_col] = data[component].rolling(window).apply(
                    lambda x: np.sum(np.sign(x.iloc[:-1]) == np.sign(x.iloc[-1])), raw=False
                )
            elif window == 5 and component in ['Range_Momentum', 'Meso_Flow_Asymmetry']:
                persistence_col = f"{component.split('_')[0]}_{component.split('_')[1]}_Persistence"
                data[persistence_col] = data[component].rolling(window).apply(
                    lambda x: np.sum(np.sign(x.iloc[:-1]) == np.sign(x.iloc[-1])), raw=False
                )
            elif window == 13 and component in ['Extended_Range_Momentum', 'Macro_Flow_Asymmetry']:
                persistence_col = f"{component.split('_')[0]}_{component.split('_')[1]}_Persistence"
                data[persistence_col] = data[component].rolling(window).apply(
                    lambda x: np.sum(np.sign(x.iloc[:-1]) == np.sign(x.iloc[-1])), raw=False
                )
    
    data['Momentum_Persistence_Score'] = data['Opening_Momentum_Persistence'] * data['Range_Momentum_Persistence'] * data['Extended_Range_Persistence']
    data['Flow_Persistence_Score'] = data['Micro_Flow_Persistence'] * data['Meso_Flow_Persistence'] * data['Macro_Flow_Persistence']
    data['Fractal_Regime_Persistence'] = data['Momentum_Persistence_Score'] * data['Flow_Persistence_Score']
    
    # Hierarchical Fractal Integration
    data['Momentum_Fractal_Core'] = data['Opening_Momentum_Strength'] * data['Range_Momentum'] * data['Extended_Range_Momentum']
    data['Flow_Fractal_Core'] = data['Fractal_Flow_Cascade'] * data['Volume_Flow_Asymmetry']
    data['Velocity_Fractal_Core'] = data['Micro_Velocity_Momentum'] * data['Meso_Velocity_Momentum'] * data['Macro_Velocity_Momentum']
    
    data['Persistence_Enhanced_Momentum'] = data['Momentum_Fractal_Core'] * data['Fractal_Regime_Persistence']
    data['Flow_Enhanced_Momentum'] = data['Flow_Fractal_Core'] * data['Fractal_Flow_Cascade']
    data['Velocity_Enhanced_Momentum'] = data['Velocity_Fractal_Core'] * data['Fractal_Regime_Coherence']
    
    data['Momentum_Flow_Alignment'] = np.sign(data['Persistence_Enhanced_Momentum']) * np.sign(data['Flow_Enhanced_Momentum'])
    data['Momentum_Velocity_Alignment'] = np.sign(data['Persistence_Enhanced_Momentum']) * np.sign(data['Velocity_Enhanced_Momentum'])
    data['Fractal_Regime_Coherence_Final'] = data['Momentum_Flow_Alignment'] * data['Momentum_Velocity_Alignment']
    
    # Final Alpha Construction
    data['Momentum_Flow_Base'] = data['Persistence_Enhanced_Momentum'] * data['Flow_Enhanced_Momentum']
    data['Momentum_Velocity_Base'] = data['Velocity_Enhanced_Momentum'] * data['Fractal_Regime_Persistence']
    data['Core_Fractal_Base'] = data['Momentum_Fractal_Core'] * data['Flow_Fractal_Core']
    
    data['Core_Alpha_Base'] = data['Momentum_Flow_Base'] * data['Momentum_Velocity_Base'] * data['Core_Fractal_Base']
    data['Fractal_Enhancement'] = data['Core_Alpha_Base'] * data['Fractal_Regime_Coherence_Final']
    data['Fractal_Momentum_Acceleration'] = data['Fractal_Enhancement'] * (data['close'] - data['open'])
    
    # Final Alpha Factor
    data['Multi_Scale_Fractal_Momentum_Regime_Alpha'] = (
        data['Fractal_Momentum_Acceleration'] * 
        np.sign(data['Opening_Momentum_Strength'] + data['Range_Momentum'] + data['Extended_Range_Momentum'])
    )
    
    return data['Multi_Scale_Fractal_Momentum_Regime_Alpha']
