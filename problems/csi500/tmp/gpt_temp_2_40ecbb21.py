import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Multi-Timeframe Momentum with Volume Persistence alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum calculations
    # Ultra-Short Term (1d)
    data['Momentum_1d'] = data['close'] - data['close'].shift(1)
    data['Range_1d'] = data['high'] - data['low']
    
    # Short-Term (3d)
    data['Momentum_3d'] = data['close'] - data['close'].shift(2)
    data['Range_3d'] = (data['high'] - data['low']) + \
                      (data['high'].shift(1) - data['low'].shift(1)) + \
                      (data['high'].shift(2) - data['low'].shift(2))
    
    # Medium-Term (10d)
    data['Momentum_10d'] = data['close'] - data['close'].shift(9)
    data['Range_10d'] = 0
    for i in range(10):
        data['Range_10d'] += data['high'].shift(i) - data['low'].shift(i)
    
    # Volatility Scaling
    data['Short_Term_Vol'] = data['Range_3d'] / 3
    data['Medium_Term_Vol'] = data['Range_10d'] / 10
    data['Volatility_Ratio'] = data['Short_Term_Vol'] / data['Medium_Term_Vol']
    
    # Volatility-Scaled Momentum
    data['VSM_1d'] = data['Momentum_1d'] / data['Range_1d']
    data['VSM_3d'] = data['Momentum_3d'] / data['Range_3d']
    data['VSM_10d'] = data['Momentum_10d'] / data['Range_10d']
    
    # Volume calculations
    data['Volume_3d'] = data['volume'].rolling(window=3).sum()
    data['Volume_10d'] = data['volume'].rolling(window=10).sum()
    data['Volume_Ratio'] = data['Volume_3d'] / data['Volume_10d']
    
    # Volume Direction and Streak
    data['Volume_Change_Dir'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate volume streak
    data['Volume_Streak'] = 0
    streak = 0
    for i in range(1, len(data)):
        if data['Volume_Change_Dir'].iloc[i] == data['Volume_Change_Dir'].iloc[i-1]:
            streak += 1
        else:
            streak = 0
        data.loc[data.index[i], 'Volume_Streak'] = streak
    
    data['Persistence_Strength'] = data['Volume_Streak'] * abs(data['volume'] - data['volume'].shift(1))
    
    # Volume-Momentum Alignment
    data['Alignment'] = np.sign(data['Momentum_1d']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Calculate alignment streak
    data['Alignment_Streak'] = 0
    align_streak = 0
    for i in range(1, len(data)):
        if data['Alignment'].iloc[i] > 0 and data['Alignment'].iloc[i-1] > 0:
            align_streak += 1
        else:
            align_streak = 0
        data.loc[data.index[i], 'Alignment_Streak'] = align_streak
    
    data['Alignment_Confidence'] = data['Alignment_Streak'] * abs(data['Momentum_1d'])
    
    # Base Momentum Signal
    data['Weighted_VSM'] = (4 * data['VSM_1d'] + 3 * data['VSM_3d'] + data['VSM_10d']) / 8
    data['Volume_Integrated'] = data['Weighted_VSM'] * np.log(data['volume'] + 1)
    
    # Persistence Enhancement
    data['Volume_Boost'] = data['Volume_Integrated'] * (1 + data['Volume_Streak'] / 8)
    data['Alignment_Boost'] = data['Volume_Boost'] * (1 + data['Alignment_Streak'] / 6)
    
    # Regime Adaptation
    # Volatility Scaling
    vol_scaling = np.ones(len(data))
    vol_scaling[data['Volatility_Ratio'] > 1.15] = 0.6    # High Vol
    vol_scaling[(data['Volatility_Ratio'] >= 0.85) & (data['Volatility_Ratio'] <= 1.15)] = 1.0  # Normal Vol
    vol_scaling[data['Volatility_Ratio'] < 0.85] = 1.4    # Low Vol
    
    # Volume Scaling
    vol_scale = np.ones(len(data))
    vol_scale[data['Volume_Ratio'] > 1.1] = 1.3    # High Volume
    vol_scale[(data['Volume_Ratio'] >= 0.9) & (data['Volume_Ratio'] <= 1.1)] = 1.0  # Normal Volume
    vol_scale[data['Volume_Ratio'] < 0.9] = 0.7    # Low Volume
    
    # Momentum Acceleration
    data['Acceleration'] = data['VSM_3d'] - data['VSM_10d']
    acceleration_confirmation = 1 + 0.15 * np.sign(data['Acceleration'])
    
    # Final Composite Score
    data['Raw_Factor'] = data['Alignment_Boost'] * vol_scaling * vol_scale * acceleration_confirmation
    
    # Return the final factor values
    return data['Raw_Factor']
