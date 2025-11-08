import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Reversal-Volume Regime Factor
    Combines multi-horizon reversal signals with volume confirmation and volatility regime detection
    """
    df = data.copy()
    
    # Core Reversal Signal
    # Multi-horizon Returns
    df['Ret_1d'] = df['close'] / df['close'].shift(1) - 1
    df['Ret_3d'] = df['close'] / df['close'].shift(3) - 1
    df['Ret_5d'] = df['close'] / df['close'].shift(5) - 1
    
    # Reversal Strength
    df['Rev_Short'] = -df['Ret_1d']
    df['Mom_Decay'] = -(df['Ret_3d'] - df['Ret_1d'])
    df['Composite_Rev'] = df['Rev_Short'] + 0.7 * df['Mom_Decay']
    
    # Volatility Regime Detection
    # Volatility Measures
    df['Range_Vol'] = (df['high'] - df['low']) / df['close']
    df['Price_Vol'] = df['Ret_1d'].rolling(window=3, min_periods=1).std()
    df['Total_Vol'] = df['Range_Vol'] * df['Price_Vol']
    
    # Adaptive Thresholds
    vol_percentile_20 = df['Total_Vol'].rolling(window=10, min_periods=1).apply(lambda x: np.percentile(x, 20))
    vol_percentile_10 = df['Total_Vol'].rolling(window=10, min_periods=1).apply(lambda x: np.percentile(x, 10))
    
    df['High_Vol'] = df['Total_Vol'] > vol_percentile_20
    df['Medium_Vol'] = (df['Total_Vol'] > vol_percentile_10) & (~df['High_Vol'])
    df['Low_Vol'] = ~(df['High_Vol'] | df['Medium_Vol'])
    
    # Volume Confirmation Logic
    # Volume Dynamics
    df['Vol_Mom'] = df['volume'] / df['volume'].shift(1) - 1
    df['Vol_Persist'] = df['volume'] / df['volume'].shift(3) - 1
    df['Vol_Stable'] = 1 / (abs(df['Vol_Mom']) + 0.01)
    
    # Price-Volume Alignment
    df['Dir_Align'] = np.sign(df['Ret_1d']) * np.sign(df['Vol_Mom'])
    df['Str_Align'] = abs(df['Ret_1d']) * abs(df['Vol_Mom'])
    df['Vol_Score'] = df['Dir_Align'] * df['Str_Align'] * df['Vol_Stable']
    
    # Nonlinear Transformations
    # Reversal Enhancement
    df['Rev_Scaled'] = np.tanh(df['Composite_Rev'] * 10)
    df['Rev_Signed'] = np.sign(df['Composite_Rev']) * abs(df['Rev_Scaled'])
    
    # Volume Enhancement
    df['Vol_Enhanced'] = 2 / (1 + np.exp(-df['Vol_Score'])) - 1
    df['Vol_Bounded'] = np.clip(df['Vol_Enhanced'], -1, 1)
    
    # Volatility Scaling
    df['Vol_Adj'] = 1 / (df['Total_Vol'] + 0.001)
    df['Scaled_Factor'] = df['Rev_Signed'] * df['Vol_Bounded'] * df['Vol_Adj']
    
    # Regime-Adaptive Integration
    # Regime Multipliers
    df['Factor_High'] = df['Scaled_Factor'] * 1.5
    df['Factor_Medium'] = df['Scaled_Factor'] * 1.1
    df['Factor_Low'] = df['Scaled_Factor'] * 0.8
    
    # Final Selection
    conditions = [
        df['High_Vol'],
        df['Medium_Vol'],
        df['Low_Vol']
    ]
    choices = [
        df['Factor_High'],
        df['Factor_Medium'],
        df['Factor_Low']
    ]
    
    df['Adaptive_Reversal_Volume_Factor'] = np.select(conditions, choices, default=df['Scaled_Factor'])
    
    return df['Adaptive_Reversal_Volume_Factor']
