import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize EMA columns
    df = df.copy()
    df['EMA_range'] = df['high'] - df['low']
    df['EMA_vol_chg'] = 0.0
    
    # Calculate EMA_range and EMA_vol_chg
    for i in range(1, len(df)):
        if i == 1:
            df.iloc[i, df.columns.get_loc('EMA_range')] = 0.15 * (df.iloc[i]['high'] - df.iloc[i]['low']) + 0.85 * (df.iloc[i-1]['high'] - df.iloc[i-1]['low'])
            df.iloc[i, df.columns.get_loc('EMA_vol_chg')] = 0.15 * abs(df.iloc[i]['volume'] / df.iloc[i-1]['volume'] - 1)
        else:
            df.iloc[i, df.columns.get_loc('EMA_range')] = 0.15 * (df.iloc[i]['high'] - df.iloc[i]['low']) + 0.85 * df.iloc[i-1]['EMA_range']
            df.iloc[i, df.columns.get_loc('EMA_vol_chg')] = 0.15 * abs(df.iloc[i]['volume'] / df.iloc[i-1]['volume'] - 1) + 0.85 * df.iloc[i-1]['EMA_vol_chg']
    
    # Combined Volatility Score
    df['Vol_Score'] = df['EMA_range'] * df['EMA_vol_chg']
    
    # Timeframe Weights
    df['Fast_Weight'] = np.exp(-df['Vol_Score'] / 0.01)
    df['Medium_Weight'] = np.exp(-df['Vol_Score'] / 0.02)
    df['Slow_Weight'] = np.exp(-df['Vol_Score'] / 0.04)
    
    # Price Momentum Ratios
    df['Fast_Price_Ratio'] = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(window=4).max() - df['low'].rolling(window=4).min() + 0.0001)
    df['Medium_Price_Ratio'] = (df['close'] - df['close'].shift(8)) / (df['high'].rolling(window=9).max() - df['low'].rolling(window=9).min() + 0.0001)
    df['Slow_Price_Ratio'] = (df['close'] - df['close'].shift(21)) / (df['high'].rolling(window=22).max() - df['low'].rolling(window=22).min() + 0.0001)
    
    # Volume Momentum Ratios
    df['Fast_Volume_Ratio'] = (df['volume'] - df['volume'].shift(3)) / (df['volume'].rolling(window=4).max() - df['volume'].rolling(window=4).min() + 0.0001)
    df['Medium_Volume_Ratio'] = (df['volume'] - df['volume'].shift(8)) / (df['volume'].rolling(window=9).max() - df['volume'].rolling(window=9).min() + 0.0001)
    df['Slow_Volume_Ratio'] = (df['volume'] - df['volume'].shift(21)) / (df['volume'].rolling(window=22).max() - df['volume'].rolling(window=22).min() + 0.0001)
    
    # Independent Volatility Scaling
    df['Price_Vol_Normalizer'] = 1 / (df['EMA_range'] + 0.0001)
    df['Volume_Vol_Normalizer'] = 1 / (df['EMA_vol_chg'] + 0.0001)
    
    # Scaled Divergence Components
    df['Fast_Divergence'] = (df['Fast_Price_Ratio'] * df['Price_Vol_Normalizer']) - (df['Fast_Volume_Ratio'] * df['Volume_Vol_Normalizer'])
    df['Medium_Divergence'] = (df['Medium_Price_Ratio'] * df['Price_Vol_Normalizer']) - (df['Medium_Volume_Ratio'] * df['Volume_Vol_Normalizer'])
    df['Slow_Divergence'] = (df['Slow_Price_Ratio'] * df['Price_Vol_Normalizer']) - (df['Slow_Volume_Ratio'] * df['Volume_Vol_Normalizer'])
    
    # Regime-Weighted Blending
    df['Fast_Component'] = df['Fast_Divergence'] * df['Fast_Weight']
    df['Medium_Component'] = df['Medium_Divergence'] * df['Medium_Weight']
    df['Slow_Component'] = df['Slow_Divergence'] * df['Slow_Weight']
    
    # Convergence Signal Enhancement
    df['Component_Alignment'] = np.sign(df['Fast_Component']) * np.sign(df['Medium_Component']) * np.sign(df['Slow_Component'])
    df['Signal_Strength'] = abs(df['Fast_Component']) + abs(df['Medium_Component']) + abs(df['Slow_Component'])
    
    # Final Alpha Output
    alpha = df['Component_Alignment'] * df['Signal_Strength']
    
    return alpha
