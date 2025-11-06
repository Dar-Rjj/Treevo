import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Scale Momentum
    data['Micro_Momentum'] = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 0.001)
    data['Meso_Momentum'] = (data['close'] - data['close'].shift(3)) / (data['close'].shift(3) + 0.001)
    data['Macro_Momentum'] = (data['close'] - data['close'].shift(8)) / (data['close'].shift(8) + 0.001)
    
    # Acceleration Quality
    data['Micro_Accel'] = data['Micro_Momentum'] - data['Micro_Momentum'].shift(1)
    data['Meso_Accel'] = data['Meso_Momentum'] - data['Meso_Momentum'].shift(2)
    data['Macro_Accel'] = data['Macro_Momentum'] - data['Macro_Momentum'].shift(5)
    
    data['Acceleration_Consistency'] = (np.sign(data['Micro_Accel']) * 
                                      np.sign(data['Meso_Accel']) * 
                                      np.sign(data['Macro_Accel']))
    
    data['Acceleration_Magnitude'] = (np.abs(data['Micro_Accel']) * 
                                    np.abs(data['Meso_Accel']) * 
                                    np.abs(data['Macro_Accel']))
    
    # Volatility Transmission
    data['Micro_Volatility'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                               (np.minimum(data['open'], data['close']) - data['low'] + 0.001))
    
    # Rolling calculations for Meso Volatility
    data['High_3d'] = data['high'].rolling(window=3, min_periods=1).max()
    data['Low_3d'] = data['low'].rolling(window=3, min_periods=1).min()
    data['Meso_Volatility'] = ((data['High_3d'] - np.maximum(data['open'], data['close'])) / 
                              (np.minimum(data['open'], data['close']) - data['Low_3d'] + 0.001))
    
    # Momentum-Volatility Divergence
    divergence_conditions = [
        np.sign(data['Micro_Momentum']) != np.sign(data['Micro_Volatility']),
        np.sign(data['Meso_Momentum']) != np.sign(data['Meso_Volatility']),
        np.sign(data['Macro_Momentum']) != np.sign(data['Micro_Volatility'] * data['Meso_Volatility'])
    ]
    data['Divergence_Count'] = sum(condition.astype(int) for condition in divergence_conditions)
    
    data['Volatility_Weighted_Momentum'] = (data['Divergence_Count'] * 
                                          (data['high'] - data['low']) / 
                                          (data['high'].shift(1) - data['low'].shift(1) + 0.001))
    
    # Volume-Pressure Dynamics
    data['Pressure_Differential'] = ((data['high'] - data['open']) * (data['close'] - data['low']) - 
                                   (data['open'] - data['low']) * (data['high'] - data['close']))
    
    data['Volume_Pressure'] = (data['volume'] * data['Pressure_Differential'] / 
                             (data['volume'].shift(1) * data['Pressure_Differential'].shift(1) + 0.001))
    
    # Amount-Efficiency Framework
    data['Efficiency_Divergence'] = (np.abs((data['high'] - data['open']) - (data['open'] - data['low'])) / 
                                   (data['high'] - data['low'] + 0.001))
    
    data['Trade_Quality'] = (data['Efficiency_Divergence'] * 
                           (data['amount'] / (data['volume'] + 0.001)) * 
                           np.sign(data['Micro_Momentum'] + data['Meso_Momentum'] + data['Macro_Momentum']))
    
    # Fractal Persistence - Momentum Persistence
    micro_persistence = []
    meso_persistence = []
    macro_persistence = []
    
    for i in range(len(data)):
        if i >= 2:
            micro_count = sum(np.sign(data['Micro_Momentum'].iloc[i-j]) == np.sign(data['Micro_Momentum'].iloc[i]) 
                            for j in range(1, 3) if i-j >= 0)
        else:
            micro_count = 0
            
        if i >= 3:
            meso_count = sum(np.sign(data['Meso_Momentum'].iloc[i-j]) == np.sign(data['Meso_Momentum'].iloc[i]) 
                           for j in range(1, 4) if i-j >= 0)
        else:
            meso_count = 0
            
        if i >= 5:
            macro_count = sum(np.sign(data['Macro_Momentum'].iloc[i-j]) == np.sign(data['Macro_Momentum'].iloc[i]) 
                            for j in range(1, 6) if i-j >= 0)
        else:
            macro_count = 0
            
        micro_persistence.append(micro_count)
        meso_persistence.append(meso_count)
        macro_persistence.append(macro_count)
    
    data['Persistence_Score'] = (pd.Series(micro_persistence, index=data.index) * 
                               pd.Series(meso_persistence, index=data.index) * 
                               pd.Series(macro_persistence, index=data.index))
    
    # Acceleration Persistence
    micro_accel_pers = []
    meso_accel_pers = []
    macro_accel_pers = []
    
    for i in range(len(data)):
        if i >= 2:
            micro_acc_count = sum(np.sign(data['Micro_Momentum'].iloc[i-j] - data['Micro_Momentum'].iloc[i-j-1]) == 
                                np.sign(data['Micro_Momentum'].iloc[i] - data['Micro_Momentum'].iloc[i-1]) 
                                for j in range(1, 3) if i-j-1 >= 0)
        else:
            micro_acc_count = 0
            
        if i >= 3:
            meso_acc_count = sum(np.sign(data['Meso_Momentum'].iloc[i-j] - data['Meso_Momentum'].iloc[i-j-2]) == 
                               np.sign(data['Meso_Momentum'].iloc[i] - data['Meso_Momentum'].iloc[i-2]) 
                               for j in range(1, 4) if i-j-2 >= 0)
        else:
            meso_acc_count = 0
            
        if i >= 5:
            macro_acc_count = sum(np.sign(data['Macro_Momentum'].iloc[i-j] - data['Macro_Momentum'].iloc[i-j-5]) == 
                                np.sign(data['Macro_Momentum'].iloc[i] - data['Macro_Momentum'].iloc[i-5]) 
                                for j in range(1, 6) if i-j-5 >= 0)
        else:
            macro_acc_count = 0
            
        micro_accel_pers.append(micro_acc_count)
        meso_accel_pers.append(meso_acc_count)
        macro_accel_pers.append(macro_acc_count)
    
    data['Accel_Persistence'] = (pd.Series(micro_accel_pers, index=data.index) * 
                               pd.Series(meso_accel_pers, index=data.index) * 
                               pd.Series(macro_accel_pers, index=data.index))
    
    # Final Alpha Construction
    data['Momentum_Core'] = data['Acceleration_Consistency'] * data['Acceleration_Magnitude'] * data['Persistence_Score']
    data['Volatility_Core'] = data['Volatility_Weighted_Momentum'] * data['Volume_Pressure']
    data['Efficiency_Core'] = data['Trade_Quality'] * data['Accel_Persistence']
    
    data['Final_Alpha'] = (data['Momentum_Core'] * data['Volatility_Core'] * data['Efficiency_Core'] * 
                          np.sign(data['Micro_Momentum'] + data['Meso_Momentum'] + data['Macro_Momentum']))
    
    return data['Final_Alpha']
