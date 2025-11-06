import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic components
    df['Opening_Fractal_Strength'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 0.001)
    df['Closing_Fractal_Strength'] = (df['high'] - df['close']) / (df['high'] - df['low'] + 0.001)
    df['Fractal_Strength_Divergence'] = df['Opening_Fractal_Strength'] - df['Closing_Fractal_Strength']
    
    # Multi-Scale Asymmetry Cascade
    # Micro Asymmetry (1-day)
    df['price_change_micro'] = df['close'] - df['close'].shift(1)
    df['range_micro'] = df['high'] - df['low'] + 0.001
    df['Micro_Asymmetry'] = (df['price_change_micro'] / df['range_micro']) * np.sign(df['Fractal_Strength_Divergence'])
    
    # Meso Asymmetry (5-day)
    df['price_change_meso'] = df['close'] - df['close'].shift(5)
    df['high_5d'] = df['high'].rolling(window=6, min_periods=6).max()  # t-5 to t inclusive
    df['low_5d'] = df['low'].rolling(window=6, min_periods=6).min()    # t-5 to t inclusive
    df['range_meso'] = df['high_5d'] - df['low_5d'] + 0.001
    df['Meso_Asymmetry'] = (df['price_change_meso'] / df['range_meso']) * np.sign(df['Fractal_Strength_Divergence'])
    
    # Macro Asymmetry (13-day)
    df['price_change_macro'] = df['close'] - df['close'].shift(13)
    df['high_13d'] = df['high'].rolling(window=14, min_periods=14).max()  # t-13 to t inclusive
    df['low_13d'] = df['low'].rolling(window=14, min_periods=14).min()    # t-13 to t inclusive
    df['range_macro'] = df['high_13d'] - df['low_13d'] + 0.001
    df['Macro_Asymmetry'] = (df['price_change_macro'] / df['range_macro']) * np.sign(df['Fractal_Strength_Divergence'])
    
    # Volume Fractal Integration
    df['Volume_Fractal_Divergence'] = df['volume'] * (df['Opening_Fractal_Strength'] - df['Closing_Fractal_Strength'])
    
    # Volume Fractal Persistence
    df['Volume_Fractal_Positive'] = (df['Volume_Fractal_Divergence'] > 0).astype(int)
    df['Volume_Fractal_Persistence'] = 0
    
    # Calculate consecutive positive days
    for i in range(len(df)):
        if i == 0:
            df.iloc[i, df.columns.get_loc('Volume_Fractal_Persistence')] = df.iloc[i]['Volume_Fractal_Positive']
        else:
            if df.iloc[i]['Volume_Fractal_Positive'] == 1:
                df.iloc[i, df.columns.get_loc('Volume_Fractal_Persistence')] = df.iloc[i-1]['Volume_Fractal_Persistence'] + 1
            else:
                df.iloc[i, df.columns.get_loc('Volume_Fractal_Persistence')] = 0
    
    # Volatility Cluster Momentum
    df['TrueRange'] = np.maximum(df['high'] - df['low'], 
                                np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                          abs(df['low'] - df['close'].shift(1))))
    
    df['Cluster_Momentum'] = (df['TrueRange'].rolling(window=3, min_periods=3).sum() - 
                             df['TrueRange'].shift(3).rolling(window=3, min_periods=3).sum())
    
    # Gap Momentum Integration
    df['Gap_Momentum'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * \
                        (1 - abs((df['close'] - df['open']) / (df['open'] + 0.001)))
    
    # Final Alpha Construction
    df['Core_Fractal_Signal'] = (df['Micro_Asymmetry'] + df['Meso_Asymmetry'] + df['Macro_Asymmetry']) * df['Cluster_Momentum']
    df['Quality_Enhanced_Fractal_Alpha'] = df['Core_Fractal_Signal'] * df['Gap_Momentum'] * df['Volume_Fractal_Persistence']
    
    # Return the final alpha factor
    return df['Quality_Enhanced_Fractal_Alpha']
