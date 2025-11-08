import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum with Volume Persistence and Volatility Regime Adaptation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Framework
    # Ultra-Short Term (1-2 days)
    data['Momentum_1d'] = data['close'] - data['close'].shift(1)
    data['Range_1d'] = data['high'] - data['low']
    data['Volume_1d'] = data['volume']
    
    # Short-Term (3-5 days)
    data['Momentum_3d'] = data['close'] - data['close'].shift(2)
    data['Range_3d'] = (data['high'] - data['low']) + (data['high'].shift(1) - data['low'].shift(1)) + (data['high'].shift(2) - data['low'].shift(2))
    data['Volume_3d'] = data['volume'] + data['volume'].shift(1) + data['volume'].shift(2)
    
    # Medium-Term (6-10 days)
    data['Momentum_10d'] = data['close'] - data['close'].shift(9)
    
    # Calculate 10-day range sum
    range_10d = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 9:
            range_10d.iloc[i] = sum(data['high'].iloc[i-j] - data['low'].iloc[i-j] for j in range(10))
        else:
            range_10d.iloc[i] = np.nan
    data['Range_10d'] = range_10d
    
    # Calculate 10-day volume sum
    volume_10d = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 9:
            volume_10d.iloc[i] = sum(data['volume'].iloc[i-j] for j in range(10))
        else:
            volume_10d.iloc[i] = np.nan
    data['Volume_10d'] = volume_10d
    
    # Volume Persistence Analysis
    data['Volume_Change_Direction'] = np.sign(data['volume'] - data['volume'].shift(1))
    
    # Volume Direction Streak
    volume_streak = pd.Series(index=data.index, dtype=float)
    current_streak = 0
    for i in range(len(data)):
        if i == 0 or np.isnan(data['Volume_Change_Direction'].iloc[i]) or np.isnan(data['Volume_Change_Direction'].iloc[i-1]):
            volume_streak.iloc[i] = 0
            current_streak = 0
        elif data['Volume_Change_Direction'].iloc[i] == data['Volume_Change_Direction'].iloc[i-1]:
            current_streak += 1
            volume_streak.iloc[i] = current_streak
        else:
            current_streak = 1
            volume_streak.iloc[i] = current_streak
    data['Volume_Direction_Streak'] = volume_streak
    
    data['Volume_Persistence_Strength'] = data['Volume_Direction_Streak'] * abs(data['volume'] - data['volume'].shift(1))
    
    # Volume-Momentum Alignment
    data['Alignment_Signal'] = np.sign(data['Momentum_1d']) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Alignment Streak
    alignment_streak = pd.Series(index=data.index, dtype=float)
    current_alignment_streak = 0
    for i in range(len(data)):
        if i == 0 or np.isnan(data['Alignment_Signal'].iloc[i]):
            alignment_streak.iloc[i] = 0
            current_alignment_streak = 0
        elif data['Alignment_Signal'].iloc[i] > 0:
            current_alignment_streak += 1
            alignment_streak.iloc[i] = current_alignment_streak
        else:
            current_alignment_streak = 0
            alignment_streak.iloc[i] = 0
    data['Alignment_Streak'] = alignment_streak
    
    data['Alignment_Confidence'] = data['Alignment_Streak'] * abs(data['Momentum_1d'])
    
    # Volume Regime Detection
    data['Volume_Ratio'] = data['Volume_3d'] / data['Volume_10d']
    data['High_Volume_Regime'] = (data['Volume_Ratio'] > 1.1).astype(int)
    data['Normal_Volume_Regime'] = ((data['Volume_Ratio'] >= 0.9) & (data['Volume_Ratio'] <= 1.1)).astype(int)
    data['Low_Volume_Regime'] = (data['Volume_Ratio'] < 0.9).astype(int)
    
    # Volatility Regime Adaptation
    data['Short_Term_Volatility'] = data['Range_3d'] / 3
    data['Medium_Term_Volatility'] = data['Range_10d'] / 10
    data['Volatility_Ratio'] = data['Short_Term_Volatility'] / data['Medium_Term_Volatility']
    
    data['High_Volatility'] = (data['Volatility_Ratio'] > 1.15).astype(int)
    data['Normal_Volatility'] = ((data['Volatility_Ratio'] >= 0.85) & (data['Volatility_Ratio'] <= 1.15)).astype(int)
    data['Low_Volatility'] = (data['Volatility_Ratio'] < 0.85).astype(int)
    
    # Volatility-Scaled Momentum
    data['VSM_1d'] = data['Momentum_1d'] / data['Range_1d']
    data['VSM_3d'] = data['Momentum_3d'] / data['Range_3d']
    data['VSM_10d'] = data['Momentum_10d'] / data['Range_10d']
    
    # Momentum Acceleration Framework
    data['Acceleration_3d'] = data['VSM_3d'] - data['VSM_10d']
    data['Acceleration_Direction'] = np.sign(data['Acceleration_3d'])
    data['Acceleration_Magnitude'] = abs(data['Acceleration_3d'])
    
    data['Acceleration_1d'] = data['VSM_1d'] - data['VSM_3d']
    data['Recent_Direction'] = np.sign(data['Acceleration_1d'])
    data['Recent_Magnitude'] = abs(data['Acceleration_1d'])
    
    # Acceleration Persistence
    acceleration_streak = pd.Series(index=data.index, dtype=float)
    current_acceleration_streak = 0
    for i in range(len(data)):
        if i == 0 or np.isnan(data['Acceleration_Direction'].iloc[i]) or np.isnan(data['Acceleration_Direction'].iloc[i-1]):
            acceleration_streak.iloc[i] = 0
            current_acceleration_streak = 0
        elif data['Acceleration_Direction'].iloc[i] == data['Acceleration_Direction'].iloc[i-1]:
            current_acceleration_streak += 1
            acceleration_streak.iloc[i] = current_acceleration_streak
        else:
            current_acceleration_streak = 1
            acceleration_streak.iloc[i] = current_acceleration_streak
    data['Acceleration_Streak'] = acceleration_streak
    
    data['Acceleration_Confidence'] = data['Acceleration_Streak'] * data['Acceleration_Magnitude']
    data['Momentum_Consistency'] = np.sign(data['VSM_1d']) * np.sign(data['VSM_3d']) * np.sign(data['VSM_10d'])
    
    # Adaptive Factor Construction
    # Base Momentum Signal
    data['Weighted_VSM'] = (4 * data['VSM_1d'] + 3 * data['VSM_3d'] + data['VSM_10d']) / 8
    data['Volume_Integration'] = data['Weighted_VSM'] * np.log(data['volume'] + 1)
    
    # Persistence Enhancement
    data['Volume_Persistence_Boost'] = data['Volume_Integration'] * (1 + data['Volume_Direction_Streak'] / 10)
    data['Alignment_Boost'] = data['Volume_Persistence_Boost'] * (1 + data['Alignment_Streak'] / 8)
    
    # Volatility Regime Adaptation
    volatility_scaling = pd.Series(index=data.index, dtype=float)
    timeframe_weights = pd.Series(index=data.index, dtype=object)
    
    for i in range(len(data)):
        if data['High_Volatility'].iloc[i] == 1:
            volatility_scaling.iloc[i] = 0.7
            timeframe_weights.iloc[i] = (6, 2, 2)
        elif data['Normal_Volatility'].iloc[i] == 1:
            volatility_scaling.iloc[i] = 1.0
            timeframe_weights.iloc[i] = (4, 3, 3)
        else:  # Low Volatility
            volatility_scaling.iloc[i] = 1.3
            timeframe_weights.iloc[i] = (2, 3, 5)
    
    data['Volatility_Scaling'] = volatility_scaling
    data['Timeframe_Weights'] = timeframe_weights
    
    # Apply timeframe-specific weighting
    data['Volatility_Adjusted_VSM'] = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if not pd.isna(data['Timeframe_Weights'].iloc[i]):
            w1, w2, w3 = data['Timeframe_Weights'].iloc[i]
            total_weight = w1 + w2 + w3
            data['Volatility_Adjusted_VSM'].iloc[i] = (
                w1 * data['VSM_1d'].iloc[i] + 
                w2 * data['VSM_3d'].iloc[i] + 
                w3 * data['VSM_10d'].iloc[i]
            ) / total_weight
    
    # Volume Regime Integration
    volume_scaling = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if data['High_Volume_Regime'].iloc[i] == 1:
            volume_scaling.iloc[i] = 1.2
        elif data['Normal_Volume_Regime'].iloc[i] == 1:
            volume_scaling.iloc[i] = 1.0
        else:  # Low Volume
            volume_scaling.iloc[i] = 0.8
    
    data['Volume_Scaling'] = volume_scaling
    
    # Volume-Persistence Interaction
    data['Volume_Persistence_Interaction'] = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        volume_persistence_score = min(data['Volume_Direction_Streak'].iloc[i] / 10, 1.0)
        if data['High_Volume_Regime'].iloc[i] == 1 and volume_persistence_score > 0.7:
            data['Volume_Persistence_Interaction'].iloc[i] = 1.3
        elif data['Low_Volume_Regime'].iloc[i] == 1 and volume_persistence_score < 0.3:
            data['Volume_Persistence_Interaction'].iloc[i] = 0.7
        else:
            data['Volume_Persistence_Interaction'].iloc[i] = 1.0
    
    # Acceleration Enhancement
    data['Acceleration_Confirmation'] = 1 + 0.2 * data['Acceleration_Direction']
    data['Acceleration_Persistence_Boost'] = 1 + data['Acceleration_Confidence'] / 5
    data['Momentum_Consistency_Bonus'] = 1 + 0.15 * data['Momentum_Consistency']
    
    # Final Alpha Output
    data['Base_Momentum_Component'] = data['Alignment_Boost']
    data['Volatility_Adjusted_Component'] = data['Volatility_Adjusted_VSM'] * data['Volatility_Scaling']
    data['Volume_Enhanced_Component'] = data['Volatility_Adjusted_Component'] * data['Volume_Scaling'] * data['Volume_Persistence_Interaction']
    data['Acceleration_Enhanced_Component'] = (
        data['Volume_Enhanced_Component'] * 
        data['Acceleration_Confirmation'] * 
        data['Acceleration_Persistence_Boost'] * 
        data['Momentum_Consistency_Bonus']
    )
    
    # Composite Alpha Value
    alpha = data['Acceleration_Enhanced_Component']
    
    # Clean up any infinite values
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    
    return alpha
