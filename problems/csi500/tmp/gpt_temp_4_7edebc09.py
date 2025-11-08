import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Scaled Momentum with Volume Persistence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Construction
    # Ultra-Short Term (1-2 days)
    data['Momentum_1d'] = data['close'] - data['close'].shift(1)
    data['Momentum_2d'] = data['close'] - data['close'].shift(2)
    data['Combined_2d'] = (data['Momentum_1d'] + data['Momentum_2d']) / 2
    
    # Short-Term (3-5 days)
    data['Momentum_3d'] = data['close'] - data['close'].shift(3)
    data['Momentum_5d'] = data['close'] - data['close'].shift(5)
    data['Combined_5d'] = (data['Momentum_3d'] + data['Momentum_5d']) / 2
    
    # Medium-Term (6-10 days)
    data['Momentum_7d'] = data['close'] - data['close'].shift(7)
    data['Momentum_10d'] = data['close'] - data['close'].shift(10)
    data['Combined_10d'] = (data['Momentum_7d'] + data['Momentum_10d']) / 2
    
    # Volatility Scaling Framework
    # Range-Based Volatility Calculation
    data['Daily_Range'] = data['high'] - data['low']
    data['Short_Term_Vol'] = (data['Daily_Range'] + data['Daily_Range'].shift(1) + data['Daily_Range'].shift(2)) / 3
    data['Medium_Term_Vol'] = data['Daily_Range'].rolling(window=10).mean()
    data['Volatility_Ratio'] = data['Short_Term_Vol'] / data['Medium_Term_Vol']
    
    # Volatility-Scaled Momentum
    data['VSM_2d'] = data['Combined_2d'] / data['Short_Term_Vol']
    data['VSM_5d'] = data['Combined_5d'] / data['Medium_Term_Vol']
    data['VSM_10d'] = data['Combined_10d'] / data['Medium_Term_Vol']
    
    # Volatility Regime Classification
    def get_volatility_multiplier(vol_ratio):
        if vol_ratio > 1.2:
            return 0.7
        elif vol_ratio >= 0.8:
            return 1.0
        else:
            return 1.3
    
    # Volume Persistence Analysis
    # Volume Direction Patterns
    data['Volume_Change'] = data['volume'] - data['volume'].shift(1)
    data['Volume_Direction'] = np.sign(data['Volume_Change'])
    
    # Calculate volume direction streak
    data['Volume_Streak'] = 0
    for i in range(1, len(data)):
        if data['Volume_Direction'].iloc[i] == data['Volume_Direction'].iloc[i-1]:
            data.loc[data.index[i], 'Volume_Streak'] = data['Volume_Streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'Volume_Streak'] = 1
    
    data['Volume_Persistence_Score'] = data['Volume_Streak'] * abs(data['Volume_Change'])
    
    # Volume-Momentum Alignment
    data['Alignment_Signal'] = np.sign(data['Combined_2d']) * np.sign(data['Volume_Change'])
    data['Alignment_Streak'] = 0
    for i in range(1, len(data)):
        if data['Alignment_Signal'].iloc[i] > 0:
            data.loc[data.index[i], 'Alignment_Streak'] = data['Alignment_Streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'Alignment_Streak'] = 0
    
    data['Alignment_Confidence'] = data['Alignment_Streak'] * abs(data['Combined_2d'])
    
    # Volume Regime Detection
    data['Short_Term_Volume'] = data['volume'] + data['volume'].shift(1) + data['volume'].shift(2)
    data['Medium_Term_Volume'] = data['volume'].rolling(window=10).sum()
    data['Volume_Ratio'] = data['Short_Term_Volume'] / data['Medium_Term_Volume']
    
    def get_volume_scaling_factor(volume_ratio):
        if volume_ratio > 1.15:
            return 1.2
        elif volume_ratio >= 0.85:
            return 1.0
        else:
            return 0.8
    
    # Adaptive Factor Integration
    # Multi-Timeframe Momentum Blend
    data['Weighted_VSM'] = (0.5 * data['VSM_2d'] + 0.3 * data['VSM_5d'] + 0.2 * data['VSM_10d'])
    
    # Volatility Regime Adaptation
    data['Volatility_Multiplier'] = data['Volatility_Ratio'].apply(get_volatility_multiplier)
    data['Volatility_Adjusted_Momentum'] = data['Weighted_VSM'] * data['Volatility_Multiplier']
    
    # Volume Persistence Enhancement
    data['Volume_Direction_Boost'] = data['Volatility_Adjusted_Momentum'] * (1 + data['Volume_Streak'] / 10)
    data['Alignment_Boost'] = data['Volume_Direction_Boost'] * (1 + data['Alignment_Streak'] / 8)
    data['Persistence_Enhanced_Momentum'] = data['Alignment_Boost']
    
    # Volume Regime Final Adjustment
    data['Volume_Scaling_Factor'] = data['Volume_Ratio'].apply(get_volume_scaling_factor)
    data['Final_Alpha'] = data['Persistence_Enhanced_Momentum'] * data['Volume_Scaling_Factor']
    
    # Return the final alpha factor series
    return data['Final_Alpha']
