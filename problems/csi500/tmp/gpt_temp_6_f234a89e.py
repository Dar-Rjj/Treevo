import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price Components
    # Multi-Period Returns
    data['R1'] = data['close'] / data['close'].shift(1) - 1
    data['R3'] = data['close'] / data['close'].shift(3) - 1
    data['R5'] = data['close'] / data['close'].shift(5) - 1
    data['R10'] = data['close'] / data['close'].shift(10) - 1
    data['R20'] = data['close'] / data['close'].shift(20) - 1
    
    # Volatility Components
    data['Daily_Range'] = data['high'] - data['low']
    data['Range_3d'] = data['Daily_Range'].rolling(window=3).sum()
    data['Range_5d'] = data['Daily_Range'].rolling(window=5).sum()
    data['Range_10d'] = data['Daily_Range'].rolling(window=10).sum()
    data['Range_20d'] = data['Daily_Range'].rolling(window=20).sum()
    
    # Volume Components
    # Volume Momentum
    data['VM1'] = data['volume'] / data['volume'].shift(1) - 1
    data['VM3'] = data['volume'] / data['volume'].shift(3) - 1
    data['VM5'] = data['volume'] / data['volume'].shift(5) - 1
    data['VM10'] = data['volume'] / data['volume'].shift(10) - 1
    data['VM20'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Volume Persistence
    volume_up_5 = pd.Series(np.zeros(len(data)), index=data.index)
    volume_down_5 = pd.Series(np.zeros(len(data)), index=data.index)
    volume_up_10 = pd.Series(np.zeros(len(data)), index=data.index)
    volume_down_10 = pd.Series(np.zeros(len(data)), index=data.index)
    
    for i in range(5, len(data)):
        window = data['volume'].iloc[i-4:i+1]
        volume_up_5.iloc[i] = (window > window.shift(1)).sum() - 1  # exclude first comparison
        volume_down_5.iloc[i] = (window < window.shift(1)).sum() - 1
    
    for i in range(10, len(data)):
        window = data['volume'].iloc[i-9:i+1]
        volume_up_10.iloc[i] = (window > window.shift(1)).sum() - 1
        volume_down_10.iloc[i] = (window < window.shift(1)).sum() - 1
    
    data['Volume_Up_Days_5'] = volume_up_5
    data['Volume_Down_Days_5'] = volume_down_5
    data['Volume_Trend_5'] = data['Volume_Up_Days_5'] - data['Volume_Down_Days_5']
    
    data['Volume_Up_Days_10'] = volume_up_10
    data['Volume_Down_Days_10'] = volume_down_10
    data['Volume_Trend_10'] = data['Volume_Up_Days_10'] - data['Volume_Down_Days_10']
    
    # Volatility Regime Classification
    data['Volatility_Ratio_Short'] = (data['Range_3d'] / 3) / (data['Range_10d'] / 10)
    data['Volatility_Ratio_Medium'] = (data['Range_5d'] / 5) / (data['Range_20d'] / 20)
    
    data['High_Volatility'] = (data['Volatility_Ratio_Short'] > 1.5) | (data['Volatility_Ratio_Medium'] > 1.3)
    data['Normal_Volatility'] = ((data['Volatility_Ratio_Short'] >= 0.8) & (data['Volatility_Ratio_Short'] <= 1.5) & 
                                (data['Volatility_Ratio_Medium'] >= 0.7) & (data['Volatility_Ratio_Medium'] <= 1.3))
    data['Low_Volatility'] = (data['Volatility_Ratio_Short'] < 0.8) & (data['Volatility_Ratio_Medium'] < 0.7)
    
    # Volume-Price Convergence Analysis
    # Directional Alignment
    data['A1'] = np.sign(data['R1']) * np.sign(data['VM1'])
    data['A3'] = np.sign(data['R3']) * np.sign(data['VM3'])
    data['A5'] = np.sign(data['R5']) * np.sign(data['VM5'])
    data['A10'] = np.sign(data['R10']) * np.sign(data['VM10'])
    data['A20'] = np.sign(data['R20']) * np.sign(data['VM20'])
    
    # Magnitude Correlation
    data['M1'] = data['R1'] * data['VM1']
    data['M3'] = data['R3'] * data['VM3']
    data['M5'] = data['R5'] * data['VM5']
    data['M10'] = data['R10'] * data['VM10']
    data['M20'] = data['R20'] * data['VM20']
    
    # Convergence Metrics
    data['Direction_Score'] = (data['A1'] + data['A3'] + data['A5'] + data['A10'] + data['A20']) / 5
    data['Magnitude_Score'] = (data['M1'] + data['M3'] + data['M5'] + data['M10'] + data['M20']) / 5
    data['Overall_Convergence'] = (data['Direction_Score'] + np.sign(data['Magnitude_Score'])) / 2
    
    # Regime-Dependent Factor Construction
    base_factor = pd.Series(np.zeros(len(data)), index=data.index)
    
    # High Volatility Regime
    high_vol_mask = data['High_Volatility']
    if high_vol_mask.any():
        primary_signal = data.loc[high_vol_mask, 'R3'] / (data.loc[high_vol_mask, 'Range_3d'] / 3)
        secondary_signal = data.loc[high_vol_mask, 'R1'] / data.loc[high_vol_mask, 'Daily_Range']
        volume_weight = 0.3 + 0.4 * data.loc[high_vol_mask, 'Overall_Convergence']
        base_factor.loc[high_vol_mask] = (0.6 * primary_signal + 0.4 * secondary_signal) * volume_weight
    
    # Normal Volatility Regime
    normal_vol_mask = data['Normal_Volatility']
    if normal_vol_mask.any():
        primary_signal = data.loc[normal_vol_mask, 'R5'] / (data.loc[normal_vol_mask, 'Range_5d'] / 5)
        secondary_signal = data.loc[normal_vol_mask, 'R10'] / (data.loc[normal_vol_mask, 'Range_10d'] / 10)
        volume_weight = 0.5 + 0.3 * data.loc[normal_vol_mask, 'Overall_Convergence']
        base_factor.loc[normal_vol_mask] = (0.5 * primary_signal + 0.5 * secondary_signal) * volume_weight
    
    # Low Volatility Regime
    low_vol_mask = data['Low_Volatility']
    if low_vol_mask.any():
        primary_signal = data.loc[low_vol_mask, 'R10'] / (data.loc[low_vol_mask, 'Range_10d'] / 10)
        secondary_signal = data.loc[low_vol_mask, 'R20'] / (data.loc[low_vol_mask, 'Range_20d'] / 20)
        volume_weight = 0.7 + 0.2 * data.loc[low_vol_mask, 'Overall_Convergence']
        base_factor.loc[low_vol_mask] = (0.4 * primary_signal + 0.6 * secondary_signal) * volume_weight
    
    # Volume Persistence Enhancement
    data['Short_Term_Persistence'] = data['Volume_Trend_5'] / 5
    data['Medium_Term_Persistence'] = data['Volume_Trend_10'] / 10
    data['Persistence_Score'] = (data['Short_Term_Persistence'] + data['Medium_Term_Persistence']) / 2
    enhanced_factor = base_factor * (1 + data['Persistence_Score'])
    
    # Final Alpha Signal
    volatility_adjustment = 1 / data['Daily_Range']
    alpha_factor = enhanced_factor * volatility_adjustment
    
    # Clean up any infinite values
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    
    return alpha_factor
