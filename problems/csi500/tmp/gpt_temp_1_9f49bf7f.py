import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum with Volatility-Volume Regime Alignment alpha factor
    """
    # Price Momentum Components
    df = df.copy()
    
    # Ultra-Short Term (1-2 days)
    df['Momentum_1d'] = df['close'] - df['close'].shift(1)
    df['Range_1d'] = df['high'] - df['low']
    
    # Short-Term (3-5 days)
    df['Momentum_3d'] = df['close'] - df['close'].shift(2)
    df['Range_3d'] = (df['high'] - df['low']) + \
                     (df['high'].shift(1) - df['low'].shift(1)) + \
                     (df['high'].shift(2) - df['low'].shift(2))
    
    # Medium-Term (6-10 days)
    df['Momentum_10d'] = df['close'] - df['close'].shift(9)
    df['Range_10d'] = 0
    for i in range(10):
        df['Range_10d'] += df['high'].shift(i) - df['low'].shift(i)
    
    # Volatility Regime Framework
    df['Short_Term_Vol'] = df['Range_3d'] / 3
    df['Medium_Term_Vol'] = df['Range_10d'] / 10
    df['Volatility_Ratio'] = df['Short_Term_Vol'] / df['Medium_Term_Vol']
    
    # Volatility-Scaled Momentum
    df['VSM_1d'] = df['Momentum_1d'] / df['Range_1d']
    df['VSM_3d'] = df['Momentum_3d'] / df['Range_3d']
    df['VSM_10d'] = df['Momentum_10d'] / df['Range_10d']
    
    # Volume Persistence Framework
    df['Volume_Change'] = df['volume'] - df['volume'].shift(1)
    df['Volume_Direction'] = np.sign(df['Volume_Change'])
    
    # Calculate Volume Streak
    df['Volume_Streak'] = 0
    streak = 0
    for i in range(1, len(df)):
        if df['Volume_Direction'].iloc[i] == df['Volume_Direction'].iloc[i-1]:
            streak += 1
        else:
            streak = 0
        df.loc[df.index[i], 'Volume_Streak'] = streak
    
    # Volume-Momentum Alignment
    df['Alignment_Signal'] = np.sign(df['Momentum_1d']) * df['Volume_Direction']
    df['Alignment_Streak'] = 0
    align_streak = 0
    for i in range(1, len(df)):
        if df['Alignment_Signal'].iloc[i] > 0:
            align_streak += 1
        else:
            align_streak = 0
        df.loc[df.index[i], 'Alignment_Streak'] = align_streak
    
    df['Alignment_Strength'] = df['Alignment_Streak'] * abs(df['Momentum_1d'])
    
    # Volume Regime Classification
    df['Volume_Ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Multi-Timeframe Momentum Alignment
    df['Direction_1d'] = np.sign(df['Momentum_1d'])
    df['Direction_3d'] = np.sign(df['Momentum_3d'])
    df['Direction_10d'] = np.sign(df['Momentum_10d'])
    
    df['Alignment_Score'] = ((df['Direction_1d'] == df['Direction_3d']).astype(int) + 
                            (df['Direction_1d'] == df['Direction_10d']).astype(int) + 
                            (df['Direction_3d'] == df['Direction_10d']).astype(int)) / 3
    
    # Momentum Magnitude Consistency
    df['Magnitude_Ratio_3d_1d'] = abs(df['Momentum_3d']) / (abs(df['Momentum_1d']) + 1e-8)
    df['Magnitude_Ratio_10d_3d'] = abs(df['Momentum_10d']) / (abs(df['Momentum_3d']) + 1e-8)
    df['Magnitude_Consistency_Score'] = (df['Magnitude_Ratio_3d_1d'] + df['Magnitude_Ratio_10d_3d']) / 2
    
    # Volatility-Adjusted Alignment
    df['VSM_Direction_1d'] = np.sign(df['VSM_1d'])
    df['VSM_Direction_3d'] = np.sign(df['VSM_3d'])
    df['VSM_Direction_10d'] = np.sign(df['VSM_10d'])
    
    df['VSM_Alignment_Score'] = ((df['VSM_Direction_1d'] == df['VSM_Direction_3d']).astype(int) + 
                                (df['VSM_Direction_1d'] == df['VSM_Direction_10d']).astype(int) + 
                                (df['VSM_Direction_3d'] == df['VSM_Direction_10d']).astype(int)) / 3
    
    # Regime-Adaptive Factor Construction
    # Base Momentum Signal - Volatility-Weighted VSM
    df['Base_VSM'] = 0.0
    
    # Volatility regime classification
    high_vol_mask = df['Volatility_Ratio'] > 1.15
    low_vol_mask = df['Volatility_Ratio'] < 0.85
    normal_vol_mask = ~high_vol_mask & ~low_vol_mask
    
    # Volatility-weighted VSM blends
    df.loc[high_vol_mask, 'Base_VSM'] = (6 * df['VSM_1d'] + 3 * df['VSM_3d'] + df['VSM_10d']) / 10
    df.loc[normal_vol_mask, 'Base_VSM'] = (4 * df['VSM_1d'] + 4 * df['VSM_3d'] + 2 * df['VSM_10d']) / 10
    df.loc[low_vol_mask, 'Base_VSM'] = (2 * df['VSM_1d'] + 4 * df['VSM_3d'] + 4 * df['VSM_10d']) / 10
    
    # Volume-Enhanced Base
    df['Volume_Enhanced_Base'] = df['Base_VSM'] * (1 + np.log(abs(df['Volume_Change']) + 1))
    
    # Momentum Alignment Enhancement
    df['Direction_Aligned_Base'] = df['Volume_Enhanced_Base'] * (1 + df['Alignment_Score'] / 3)
    df['Magnitude_Consistent_Base'] = df['Direction_Aligned_Base'] * df['Magnitude_Consistency_Score']
    
    # Volume Persistence Integration
    df['Volume_Streak_Scaled'] = df['Magnitude_Consistent_Base'].copy()
    
    # Volume streak scaling
    strong_persistence = df['Volume_Streak'] >= 3
    moderate_persistence = df['Volume_Streak'] == 2
    weak_persistence = df['Volume_Streak'] <= 1
    
    df.loc[strong_persistence, 'Volume_Streak_Scaled'] *= 1.4
    df.loc[moderate_persistence, 'Volume_Streak_Scaled'] *= 1.2
    df.loc[weak_persistence, 'Volume_Streak_Scaled'] *= 1.0
    
    # Volume-Momentum Confirmation
    high_confirmation = df['Alignment_Streak'] >= 2
    moderate_confirmation = df['Alignment_Streak'] == 1
    low_confirmation = df['Alignment_Streak'] == 0
    
    df.loc[high_confirmation, 'Volume_Streak_Scaled'] *= 1.3
    df.loc[moderate_confirmation, 'Volume_Streak_Scaled'] *= 1.1
    df.loc[low_confirmation, 'Volume_Streak_Scaled'] *= 1.0
    
    # Volatility Regime Adjustment
    df['Volatility_Adjusted'] = df['Volume_Streak_Scaled'].copy()
    
    df.loc[high_vol_mask, 'Volatility_Adjusted'] *= 0.7
    df.loc[normal_vol_mask, 'Volatility_Adjusted'] *= 1.0
    df.loc[low_vol_mask, 'Volatility_Adjusted'] *= 1.3
    
    # Final Factor Integration
    df['VSM_Alignment_Multiplier'] = 1 + (df['VSM_Alignment_Score'] / 3)
    df['Final_Alpha'] = df['Volatility_Adjusted'] * df['VSM_Alignment_Multiplier']
    
    # Return the final alpha factor
    return df['Final_Alpha']
