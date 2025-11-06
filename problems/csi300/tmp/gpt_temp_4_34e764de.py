import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Fractal Efficiency
    # 10-day fractal efficiency
    df['fractal_10'] = abs(df['close'] - df['close'].shift(10)) / df['high'].rolling(window=10).apply(lambda x: sum(abs(x - df['low'].loc[x.index])), raw=False)
    
    # 20-day fractal efficiency
    df['fractal_20'] = abs(df['close'] - df['close'].shift(20)) / df['high'].rolling(window=20).apply(lambda x: sum(abs(x - df['low'].loc[x.index])), raw=False)
    
    # Volume Acceleration
    # 5-day volume momentum
    df['vol_mom_5'] = df['volume'] / df['volume'].shift(5) - 1
    
    # 10-day volume momentum
    df['vol_mom_10'] = df['volume'] / df['volume'].shift(10) - 1
    
    # Volume acceleration
    df['vol_accel'] = df['vol_mom_5'] - df['vol_mom_10']
    
    # Price Momentum Acceleration
    # 5-day price momentum
    df['price_mom_5'] = df['close'] / df['close'].shift(5) - 1
    
    # 10-day price momentum
    df['price_mom_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Price momentum acceleration
    df['price_mom_accel'] = df['price_mom_5'] - df['price_mom_10']
    
    # Combined Factor
    # Base factor
    df['base_factor'] = df['fractal_10'] * df['fractal_20']
    
    # Volume adjusted factor
    df['vol_adjusted'] = df['base_factor'] * (1 + df['vol_accel'])
    
    # Final factor
    df['final_factor'] = df['vol_adjusted'] * df['price_mom_accel']
    
    return df['final_factor']
