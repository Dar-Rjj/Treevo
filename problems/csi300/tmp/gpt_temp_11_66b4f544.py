import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Fractal Nonlinear Dynamics components
    df['Fractal_Asymmetric_Pressure'] = ((df['high'] - df['open'])**2 / (df['high'] - df['low'] + 0.001) - 
                                        (df['close'] - df['low'])**2 / (df['high'] - df['low'] + 0.001))
    
    df['Nonlinear_Fractal_Momentum'] = (((df['open'] - df['low'])**2 * (df['close'] - df['open'])**2 / (df['high'] - df['low'] + 0.001)) - 
                                       ((df['high'] - df['close'])**2 * (df['close'] - df['low'])**2 / (df['high'] - df['low'] + 0.001)))
    
    # Fractal Order Flow Persistence (5-day window)
    for i in range(len(df)):
        if i >= 4:
            window = df.iloc[i-4:i+1]
            numerator = ((window['close'] - window['open'])**3).sum()
            denominator = ((window['close'] - window['open']).abs()**2).sum() + 0.001
            df.loc[df.index[i], 'Fractal_Order_Flow_Persistence'] = (numerator / denominator) * (df.loc[df.index[i], 'high'] - df.loc[df.index[i], 'low'])**0.5
        else:
            df.loc[df.index[i], 'Fractal_Order_Flow_Persistence'] = 0
    
    # Multi-Scale Nonlinear Fractal components
    df['Micro_Nonlinear_Fractal'] = ((df['close'] - df['close'].shift(1))**2 / (df['high'] - df['low'] + 0.001) * 
                                    (df['close'] - df['close'].shift(2))**2 / (df['high'].shift(2) - df['low'].shift(2) + 0.001))
    
    # Macro Nonlinear Fractal components
    df['High_13d'] = df['high'].rolling(window=14, min_periods=1).max()
    df['Low_13d'] = df['low'].rolling(window=14, min_periods=1).min()
    df['High_21d'] = df['high'].rolling(window=22, min_periods=1).max()
    df['Low_21d'] = df['low'].rolling(window=22, min_periods=1).min()
    
    df['Macro_Nonlinear_Fractal'] = ((df['close'] - df['close'].shift(13))**3 / (df['High_13d'] - df['Low_13d'] + 0.001) * 
                                    (df['close'] - df['close'].shift(21))**3 / (df['High_21d'] - df['Low_21d'] + 0.001))
    
    df['Volatility_Fractal_Curvature'] = ((df['high'] - df['low']) - 2*(df['high'].shift(1) - df['low'].shift(1)) + 
                                         (df['high'].shift(2) - df['low'].shift(2))) * (df['close'] - df['close'].shift(2))**2
    
    # Volume-Fractal Synchronization components
    df['High_8d'] = df['high'].rolling(window=9, min_periods=1).max()
    df['Low_8d'] = df['low'].rolling(window=9, min_periods=1).min()
    
    df['Nonlinear_Volume_Flow_Fractal'] = (df['volume']**1.5 * (df['close'] - df['open'])**2 / (df['high'] - df['low'] + 0.001) * 
                                          df['volume'] / (df['volume'].shift(8) + 0.001) * 
                                          (df['close'] - df['close'].shift(8))**2 / (df['High_8d'] - df['Low_8d'] + 0.001))
    
    df['Volume_Fractal_Divergence'] = ((df['volume']**2 - df['volume'].shift(1)**2) / (df['volume'].shift(1)**2 + 0.001) - 
                                      ((df['high'] - df['close'])**2 - (df['close'] - df['low'])**2) / (df['high'].shift(1) - df['low'].shift(1) + 0.001))
    
    df['Fractal_Flow_Pressure'] = (((df['high'] - df['close'])**2 - (df['close'] - df['low'])**2) / (df['high'] - df['low'] + 0.001) * 
                                  (df['volume'] - df['volume'].shift(1))**2)
    
    # Nonlinear Regime Detection components
    df['range_sq'] = (df['high'] - df['low'])**2
    df['range_sq_prev'] = df['range_sq'].shift(1)
    
    def count_range_persistence(row_idx):
        if row_idx < 3:
            return 0
        current_data = df.iloc[row_idx-2:row_idx+1]
        count_greater = (current_data['range_sq'] > current_data['range_sq_prev']).sum()
        count_less = (current_data['range_sq'] < current_data['range_sq_prev']).sum()
        return count_greater - count_less
    
    df['Fractal_Range_Persistence'] = [count_range_persistence(i) for i in range(len(df))]
    
    df['Volatility_Expansion_Strength'] = ((df['range_sq'] - df['range_sq'].shift(2)) / (df['high'].shift(2) - df['low'].shift(2) + 0.001) * 
                                          (df['close'] - df['close'].shift(1))**2)
    
    df['Nonlinear_Gap_Momentum'] = ((df['open'] - df['close'].shift(1))**3 / (df['close'].shift(1) + 0.001) * 
                                   (df['volume']**2 - df['volume'].shift(1)**2) / (df['volume']**2 + df['volume'].shift(1)**2 + 0.001))
    
    # Adaptive Nonlinear Fractal Alpha construction
    df['Expanding_Fractal_Regime'] = (df['Nonlinear_Volume_Flow_Fractal'] * df['Fractal_Asymmetric_Pressure'] * 
                                     df['Micro_Nonlinear_Fractal'] * df['Fractal_Range_Persistence'])
    
    df['Contracting_Fractal_Regime'] = (df['Nonlinear_Fractal_Momentum'] * df['Macro_Nonlinear_Fractal'] * 
                                       df['Fractal_Range_Persistence'] * df['Volatility_Fractal_Curvature'])
    
    df['Nonlinear_Flow_Score'] = (df['Fractal_Order_Flow_Persistence'] * df['Fractal_Flow_Pressure'] * 
                                 df['Volume_Fractal_Divergence'])
    
    df['Base_Nonlinear_Fractal'] = (np.sign(df['Expanding_Fractal_Regime']) * np.abs(df['Expanding_Fractal_Regime'])**0.6 * 
                                   np.sign(df['Contracting_Fractal_Regime']) * np.abs(df['Contracting_Fractal_Regime'])**0.7 * 
                                   np.sign(df['Nonlinear_Flow_Score']) * np.abs(df['Nonlinear_Flow_Score'])**0.5)
    
    df['Regime_Multiplier'] = df['Volatility_Expansion_Strength'] * df['Nonlinear_Gap_Momentum']
    
    # Final Alpha calculation
    result = (df['Base_Nonlinear_Fractal'] * (1 + np.abs(df['Regime_Multiplier'])**1.1) * 
             np.sign(df['Fractal_Range_Persistence'])**2)
    
    # Clean up intermediate columns
    cols_to_drop = ['Fractal_Asymmetric_Pressure', 'Nonlinear_Fractal_Momentum', 'Fractal_Order_Flow_Persistence',
                   'Micro_Nonlinear_Fractal', 'Macro_Nonlinear_Fractal', 'Volatility_Fractal_Curvature',
                   'Nonlinear_Volume_Flow_Fractal', 'Volume_Fractal_Divergence', 'Fractal_Flow_Pressure',
                   'Fractal_Range_Persistence', 'Volatility_Expansion_Strength', 'Nonlinear_Gap_Momentum',
                   'Expanding_Fractal_Regime', 'Contracting_Fractal_Regime', 'Nonlinear_Flow_Score',
                   'Base_Nonlinear_Fractal', 'Regime_Multiplier', 'range_sq', 'range_sq_prev',
                   'High_13d', 'Low_13d', 'High_21d', 'Low_21d', 'High_8d', 'Low_8d']
    
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return result
