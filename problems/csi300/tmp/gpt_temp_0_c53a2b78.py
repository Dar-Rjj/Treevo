import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Add small constant to avoid division by zero
    eps = 0.001
    
    # Micro-Scale Volatility
    data['Opening_Volatility_Intensity'] = ((data['high'] - data['low']) / (data['open'] + eps)) * (data['volume'] / data['volume'].shift(1))
    data['High_Volatility_Pressure'] = ((data['high'] - data['open']) / (data['close'].shift(1) + eps)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['Low_Volatility_Support'] = ((data['open'] - data['low']) / (data['close'].shift(1) + eps)) * np.sign(data['volume'].shift(1) - data['volume'])
    
    # Meso-Scale Volatility (5-day window)
    data['Range_Volatility'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / (data['close'].shift(5) + eps)
    data['High_Pressure_Volatility'] = (data['high'].rolling(window=5).max() - data['open']) / (data['low'].rolling(window=5).min() + eps)
    data['Low_Support_Volatility'] = (data['open'] - data['low'].rolling(window=5).min()) / (data['high'].rolling(window=5).max() + eps)
    
    # Macro-Scale Volatility (21-day window)
    data['Extended_Range_Volatility'] = (data['high'].rolling(window=21).max() - data['low'].rolling(window=21).min()) / (data['close'].shift(21) + eps)
    data['High_Pressure_Macro_Vol'] = (data['high'].rolling(window=21).max() - data['open']) / (data['low'].rolling(window=21).min() + eps)
    data['Low_Support_Macro_Vol'] = (data['open'] - data['low'].rolling(window=21).min()) / (data['high'].rolling(window=21).max() + eps)
    
    # Volatility Asymmetry Patterns
    data['Micro_Volatility_Asymmetry'] = ((data['high'] - data['open']) / (data['open'] - data['low'] + eps)) - ((data['close'] - data['low']) / (data['high'] - data['open'] + eps))
    data['Meso_Volatility_Asymmetry'] = ((data['high'].rolling(window=5).max() - data['open']) / (data['open'] - data['low'].rolling(window=5).min() + eps)) - ((data['open'] - data['low'].rolling(window=5).min()) / (data['high'].rolling(window=5).max() - data['open'] + eps))
    data['Macro_Volatility_Asymmetry'] = ((data['high'].rolling(window=21).max() - data['open']) / (data['open'] - data['low'].rolling(window=21).min() + eps)) - ((data['open'] - data['low'].rolling(window=21).min()) / (data['high'].rolling(window=21).max() - data['open'] + eps))
    data['Volatility_Cascade'] = data['Micro_Volatility_Asymmetry'] * data['Meso_Volatility_Asymmetry'] * data['Macro_Volatility_Asymmetry']
    
    # Amount-Volatility Regime Integration
    data['Morning_Amount_Flow'] = (data['high'] - data['open']) * data['amount'] / (data['high'] - data['low'] + eps)
    data['Afternoon_Amount_Flow'] = (data['close'] - data['low']) * data['amount'] / (data['high'] - data['low'] + eps)
    data['Amount_Flow_Asymmetry'] = data['Morning_Amount_Flow'] - data['Afternoon_Amount_Flow']
    
    data['Micro_Volatility_Amount'] = data['Opening_Volatility_Intensity'] * data['Amount_Flow_Asymmetry']
    data['Meso_Volatility_Amount'] = data['Range_Volatility'] * data['Amount_Flow_Asymmetry']
    data['Macro_Volatility_Amount'] = data['Extended_Range_Volatility'] * data['Amount_Flow_Asymmetry']
    
    data['Volatility_Consistency'] = np.sign(data['Opening_Volatility_Intensity']) * np.sign(data['Range_Volatility']) * np.sign(data['Extended_Range_Volatility'])
    data['Asymmetry_Consistency'] = np.sign(data['Micro_Volatility_Asymmetry']) * np.sign(data['Meso_Volatility_Asymmetry']) * np.sign(data['Macro_Volatility_Asymmetry'])
    data['Volatility_Regime_Coherence'] = data['Volatility_Consistency'] * data['Asymmetry_Consistency']
    
    # Volume-Density Fractal Patterns
    data['Opening_Trade_Density'] = (data['close'] - data['open']) * data['volume'] / (data['amount'] + eps)
    data['High_Trade_Density'] = (data['high'] - data['open']) * data['volume'] / (data['amount'] + eps)
    data['Low_Trade_Density'] = (data['open'] - data['low']) * data['volume'] / (data['amount'] + eps)
    
    data['Micro_Density_Asymmetry'] = data['Opening_Trade_Density'] - data['High_Trade_Density']
    data['Meso_Density_Asymmetry'] = data['Opening_Trade_Density'] - data['Low_Trade_Density']
    data['Macro_Density_Asymmetry'] = data['High_Trade_Density'] - data['Low_Trade_Density']
    
    data['Micro_Density_Volatility'] = data['Micro_Density_Asymmetry'] * data['Opening_Volatility_Intensity']
    data['Meso_Density_Volatility'] = data['Meso_Density_Asymmetry'] * data['Range_Volatility']
    data['Macro_Density_Volatility'] = data['Macro_Density_Asymmetry'] * data['Extended_Range_Volatility']
    
    # Volatility Regime Persistence
    def count_persistence(series, window):
        current_sign = np.sign(series)
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= window:
                window_data = series.iloc[i-window:i]
                count = (np.sign(window_data) == current_sign.iloc[i]).sum()
                persistence.iloc[i] = count
            else:
                persistence.iloc[i] = 0
        return persistence
    
    data['Micro_Volatility_Persistence'] = count_persistence(data['Opening_Volatility_Intensity'], 2)
    data['Meso_Volatility_Persistence'] = count_persistence(data['Range_Volatility'], 5)
    data['Macro_Volatility_Persistence'] = count_persistence(data['Extended_Range_Volatility'], 21)
    
    data['Micro_Asymmetry_Persistence'] = count_persistence(data['Micro_Volatility_Asymmetry'], 2)
    data['Meso_Asymmetry_Persistence'] = count_persistence(data['Meso_Volatility_Asymmetry'], 5)
    data['Macro_Asymmetry_Persistence'] = count_persistence(data['Macro_Volatility_Asymmetry'], 21)
    
    data['Volatility_Persistence_Score'] = data['Micro_Volatility_Persistence'] * data['Meso_Volatility_Persistence'] * data['Macro_Volatility_Persistence']
    data['Asymmetry_Persistence_Score'] = data['Micro_Asymmetry_Persistence'] * data['Meso_Asymmetry_Persistence'] * data['Macro_Asymmetry_Persistence']
    data['Volatility_Regime_Persistence'] = data['Volatility_Persistence_Score'] * data['Asymmetry_Persistence_Score']
    
    # Hierarchical Volatility Integration
    data['Volatility_Fractal_Core'] = data['Opening_Volatility_Intensity'] * data['Range_Volatility'] * data['Extended_Range_Volatility']
    data['Amount_Fractal_Core'] = data['Volatility_Cascade'] * data['Amount_Flow_Asymmetry']
    data['Density_Fractal_Core'] = data['Micro_Density_Volatility'] * data['Meso_Density_Volatility'] * data['Macro_Density_Volatility']
    
    data['Persistence_Enhanced_Volatility'] = data['Volatility_Fractal_Core'] * data['Volatility_Regime_Persistence']
    data['Amount_Enhanced_Volatility'] = data['Amount_Fractal_Core'] * data['Volatility_Cascade']
    data['Density_Enhanced_Volatility'] = data['Density_Fractal_Core'] * data['Volatility_Regime_Coherence']
    
    data['Volatility_Amount_Alignment'] = np.sign(data['Persistence_Enhanced_Volatility']) * np.sign(data['Amount_Enhanced_Volatility'])
    data['Volatility_Density_Alignment'] = np.sign(data['Persistence_Enhanced_Volatility']) * np.sign(data['Density_Enhanced_Volatility'])
    data['Volatility_Regime_Coherence_Final'] = data['Volatility_Amount_Alignment'] * data['Volatility_Density_Alignment']
    
    # Final Alpha Construction
    data['Volatility_Amount_Base'] = data['Persistence_Enhanced_Volatility'] * data['Amount_Enhanced_Volatility']
    data['Volatility_Density_Base'] = data['Density_Enhanced_Volatility'] * data['Volatility_Regime_Persistence']
    data['Core_Volatility_Base'] = data['Volatility_Fractal_Core'] * data['Amount_Fractal_Core']
    
    data['Core_Alpha_Base'] = data['Volatility_Amount_Base'] * data['Volatility_Density_Base'] * data['Core_Volatility_Base']
    data['Volatility_Enhancement'] = data['Core_Alpha_Base'] * data['Volatility_Regime_Coherence_Final']
    data['Volatility_Acceleration'] = data['Volatility_Enhancement'] * (data['high'] - data['low'])
    
    # Final Alpha Factor
    data['Multi_Scale_Volatility_Regime_Alpha'] = data['Volatility_Acceleration'] * np.sign(data['Opening_Volatility_Intensity'] + data['Range_Volatility'] + data['Extended_Range_Volatility'])
    
    return data['Multi_Scale_Volatility_Regime_Alpha']
