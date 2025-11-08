import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum with Volume Confirmation alpha factor
    """
    df = data.copy()
    
    # Price Components
    # Daily Returns
    df['Return_1d'] = df['close'] - df['close'].shift(1)
    df['Return_3d'] = df['close'] - df['close'].shift(3)
    df['Return_10d'] = df['close'] - df['close'].shift(10)
    
    # Price Range
    df['Range_1d'] = df['high'] - df['low']
    df['Range_3d'] = (df['high'] - df['low']) + (df['high'].shift(1) - df['low'].shift(1)) + (df['high'].shift(2) - df['low'].shift(2))
    df['Range_10d'] = df['Range_1d'].rolling(window=10, min_periods=10).sum()
    
    # Volume Components
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Direction'] = np.sign(df['Volume_Change'])
    
    # Volume Streak calculation
    df['Volume_Streak'] = 0
    streak = 0
    for i in range(1, len(df)):
        if df['Volume_Direction'].iloc[i] == df['Volume_Direction'].iloc[i-1]:
            streak += 1
        else:
            streak = 0
        df.iloc[i, df.columns.get_loc('Volume_Streak')] = streak
    
    # Volume Persistence
    df['Volume_Alignment'] = np.sign(df['Return_1d']) * df['Volume_Direction']
    
    # Alignment Streak calculation
    df['Alignment_Streak'] = 0
    alignment_streak = 0
    for i in range(1, len(df)):
        if df['Volume_Alignment'].iloc[i] > 0:
            alignment_streak += 1
        else:
            alignment_streak = 0
        df.iloc[i, df.columns.get_loc('Alignment_Streak')] = alignment_streak
    
    df['Volume_Confidence'] = df['Alignment_Streak'] * abs(df['Return_1d'])
    
    # Volatility Framework
    df['Vol_3d'] = df['Range_3d'] / 3
    df['Vol_10d'] = df['Range_10d'] / 10
    df['Vol_Ratio'] = df['Vol_3d'] / df['Vol_10d']
    
    # Volatility Regimes
    df['High_Vol'] = df['Vol_Ratio'] > 1.1
    df['Normal_Vol'] = (df['Vol_Ratio'] >= 0.9) & (df['Vol_Ratio'] <= 1.1)
    df['Low_Vol'] = df['Vol_Ratio'] < 0.9
    
    # Momentum Construction
    # Volatility-Scaled Momentum
    df['VSM_1d'] = df['Return_1d'] / df['Range_1d']
    df['VSM_3d'] = df['Return_3d'] / df['Range_3d']
    df['VSM_10d'] = df['Return_10d'] / df['Range_10d']
    
    # Regime-Specific Momentum Weights
    df['High_Vol_Weights'] = (6 * df['VSM_1d'] + 3 * df['VSM_3d'] + df['VSM_10d']) / 10
    df['Normal_Vol_Weights'] = (4 * df['VSM_1d'] + 3 * df['VSM_3d'] + 3 * df['VSM_10d']) / 10
    df['Low_Vol_Weights'] = (2 * df['VSM_1d'] + 3 * df['VSM_3d'] + 5 * df['VSM_10d']) / 10
    
    # Factor Integration
    # Base Momentum Signal
    df['Base_Momentum'] = np.where(
        df['High_Vol'], df['High_Vol_Weights'],
        np.where(df['Low_Vol'], df['Low_Vol_Weights'], df['Normal_Vol_Weights'])
    )
    
    # Volume Enhancement
    df['Volume_Boost'] = 1 + (df['Volume_Streak'] * 0.08)
    df['Alignment_Boost'] = 1 + (df['Alignment_Streak'] * 0.05)
    df['Volume_Weighted_Signal'] = df['Base_Momentum'] * df['Volume_Boost'] * df['Alignment_Boost']
    
    # Regime Scaling
    df['Regime_Scale'] = np.where(
        df['High_Vol'], 0.7,
        np.where(df['Low_Vol'], 1.3, 1.0)
    )
    
    # Final Alpha
    df['Alpha_Value'] = df['Volume_Weighted_Signal'] * df['Regime_Scale']
    
    return df['Alpha_Value']
