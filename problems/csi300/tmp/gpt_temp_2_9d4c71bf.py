import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volume moving averages
    data['Volume_MA_5'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['Volume_MA_10'] = data['volume'].rolling(window=10, min_periods=10).mean()
    
    # Multi-Timeframe Momentum Convergence with Volume Weighting
    # Short-Term Momentum (3-day)
    data['momentum_3d'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_3d_vol_weighted'] = data['momentum_3d'] * (data['volume'] / data['Volume_MA_5'])
    
    # Medium-Term Momentum (8-day)
    data['momentum_8d'] = data['close'] / data['close'].shift(7) - 1
    data['momentum_8d_vol_weighted'] = data['momentum_8d'] * (data['volume'] / data['Volume_MA_10'])
    
    # Convergence Analysis with Volume Confirmation
    data['momentum_direction_alignment'] = np.sign(data['momentum_3d']) * np.sign(data['momentum_8d'])
    data['convergence_strength'] = data['momentum_3d'] * data['momentum_8d']
    data['Volume_Acceleration'] = data['Volume_MA_5'] / data['Volume_MA_10']
    
    # Apply only when both returns exceed threshold (0.01%)
    threshold = 0.0001
    valid_convergence = (np.abs(data['momentum_3d']) > threshold) & (np.abs(data['momentum_8d']) > threshold)
    data['volume_weighted_convergence'] = data['convergence_strength'] * data['Volume_Acceleration']
    data['volume_weighted_convergence'] = data['volume_weighted_convergence'].where(valid_convergence, 0)
    
    # Order Flow Anchored Position Analysis
    # Volume-Weighted Anchored Price Levels
    for i in range(5):
        data[f'High_lag_{i}'] = data['high'].shift(i)
        data[f'Low_lag_{i}'] = data['low'].shift(i)
        data[f'Volume_lag_{i}'] = data['volume'].shift(i)
    
    # Recent High Anchor (weighted by volume)
    high_cols = [f'High_lag_{i}' for i in range(5)]
    vol_cols = [f'Volume_lag_{i}' for i in range(5)]
    
    data['Recent_High_Anchor'] = 0.0
    data['Recent_Low_Anchor'] = 0.0
    
    for idx in range(len(data)):
        if idx >= 4:
            highs = [data[col].iloc[idx] for col in high_cols]
            vols = [data[col].iloc[idx] for col in vol_cols]
            total_vol = sum(vols)
            if total_vol > 0:
                data.loc[data.index[idx], 'Recent_High_Anchor'] = sum(h * v for h, v in zip(highs, vols)) / total_vol
            
            lows = [data[col].iloc[idx] for col in [f'Low_lag_{i}' for i in range(5)]]
            data.loc[data.index[idx], 'Recent_Low_Anchor'] = sum(l * v for l, v in zip(lows, vols)) / total_vol
    
    # Current Session Anchor
    data['Current_Session_Anchor'] = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
    
    # Price Position Relative to Volume-Weighted Anchors
    anchor_range = data['Recent_High_Anchor'] - data['Recent_Low_Anchor']
    data['Upper_Position'] = (data['high'] - data['Recent_High_Anchor']) / anchor_range.replace(0, np.nan)
    data['Lower_Position'] = (data['Recent_Low_Anchor'] - data['low']) / anchor_range.replace(0, np.nan)
    daily_range = data['high'] - data['low']
    data['Session_Position'] = (data['close'] - (data['Current_Session_Anchor'] / data['volume'])) / daily_range.replace(0, np.nan)
    
    # Multi-Scale Order Flow Imbalance Integration
    prev_range = data['high'].shift(1) - data['low'].shift(1)
    data['Opening_Pressure'] = ((data['open'] - data['close'].shift(1)) / prev_range.replace(0, np.nan)) * data['volume']
    data['Closing_Pressure'] = ((data['close'] - data['open']) / daily_range.replace(0, np.nan)) * data['volume']
    data['Midday_Momentum'] = ((data['high'] + data['low']) / 2 - (data['open'] + data['close']) / 2) * data['volume']
    
    # Order Flow Components
    data['Order_Flow_Components'] = data['Opening_Pressure'] + data['Closing_Pressure'] + data['Midday_Momentum']
    
    # Order Flow Convergence Analysis
    data['Very_Short_term_Flow'] = data['Order_Flow_Components'].rolling(window=2, min_periods=2).sum()
    data['Short_term_Flow'] = data['Order_Flow_Components'].rolling(window=5, min_periods=5).sum()
    
    data['Flow_Convergence'] = np.sign(data['Very_Short_term_Flow'] * data['Short_term_Flow']) * \
                              np.abs(data['Very_Short_term_Flow'] - data['Short_term_Flow'])
    
    # Volume-Weighted Price Efficiency Enhancement
    data['Range_Efficiency'] = ((data['close'] - data['open']) / daily_range.replace(0, np.nan)) * data['volume']
    data['Gap_Efficiency'] = ((data['open'] - data['close'].shift(1)) / prev_range.replace(0, np.nan)) * data['volume']
    
    # Volume Distribution Analysis
    data['Upper_Volume_Concentration'] = ((data['high'] - data['close']) / daily_range.replace(0, np.nan)) * data['volume']
    data['Lower_Volume_Concentration'] = ((data['close'] - data['low']) / daily_range.replace(0, np.nan)) * data['volume']
    data['Volume_Skew'] = data['Upper_Volume_Concentration'] - data['Lower_Volume_Concentration']
    
    # Hierarchical Factor Combination
    # Base Convergence Factor
    data['Base_Convergence_Factor'] = data['volume_weighted_convergence'] * data['Flow_Convergence']
    
    # Anchor-Based Position Scaling
    data['Support_Zone_Multiplier'] = 1.0
    data['Resistance_Zone_Multiplier'] = 1.0
    data['Neutral_Zone_Multiplier'] = 1.0
    
    support_condition = data['close'] < data['Recent_Low_Anchor']
    resistance_condition = data['close'] > data['Recent_High_Anchor']
    neutral_condition = ~support_condition & ~resistance_condition
    
    data.loc[support_condition, 'Support_Zone_Multiplier'] = 1 + np.abs(data.loc[support_condition, 'Lower_Position'])
    data.loc[resistance_condition, 'Resistance_Zone_Multiplier'] = 1 + np.abs(data.loc[resistance_condition, 'Upper_Position'])
    data.loc[neutral_condition, 'Neutral_Zone_Multiplier'] = 1 + np.abs(data.loc[neutral_condition, 'Session_Position'])
    
    data['Position_Multiplier'] = data['Support_Zone_Multiplier'].fillna(1) + \
                                 data['Resistance_Zone_Multiplier'].fillna(0) + \
                                 data['Neutral_Zone_Multiplier'].fillna(0)
    
    # Efficiency Enhancement
    data['Price_Movement_Efficiency'] = (data['Range_Efficiency'] + data['Gap_Efficiency']) / 2
    
    # Final Alpha Factor Generation
    # Composite Factor Calculation
    data['Composite_Factor'] = data['Base_Convergence_Factor'] * data['Volume_Skew'] * data['Price_Movement_Efficiency']
    data['Composite_Factor'] = data['Composite_Factor'] * data['Position_Multiplier']
    
    # Volume-Weighted Smoothing
    data['Volume_Weighted_Factor'] = data['Composite_Factor'] * data['volume']
    data['Final_Alpha'] = data['Volume_Weighted_Factor'].rolling(window=3, min_periods=3).sum() / \
                         data['volume'].rolling(window=3, min_periods=3).sum()
    
    # Clean up intermediate columns
    result = data['Final_Alpha'].copy()
    
    return result
