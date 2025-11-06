import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate True Range
    data['TrueRange'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Multi-Scale Fracture Asymmetry Dynamics
    # Fracture Acceleration Asymmetry
    data['Short_Term_Fracture'] = (data['close']/data['close'].shift(1)-1) - (data['close']/data['close'].shift(3)-1)
    data['Medium_Term_Fracture'] = (data['close']/data['close'].shift(4)-1) - (data['close']/data['close'].shift(9)-1)
    data['Fracture_Acceleration'] = data['Short_Term_Fracture'] / data['Medium_Term_Fracture']
    
    # Handle division by zero
    data['Fracture_Acceleration'] = data['Fracture_Acceleration'].replace([np.inf, -np.inf], np.nan)
    
    # Range Asymmetry Fractality
    high_low_diff = data['high'] - data['low']
    high_low_diff = high_low_diff.replace(0, np.nan)  # Avoid division by zero
    data['Range_Asymmetry_Fractality'] = (
        ((data['high'] - data['close']) / high_low_diff * data['TrueRange']) - 
        ((data['close'] - data['low']) / high_low_diff * data['TrueRange'])
    )
    
    # Volume-Volatility Fracture Coordination
    data['Volume_Scaling'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['Volatility_Scaling'] = data['TrueRange'] / data['TrueRange'].rolling(window=5).mean()
    data['Volume_Volatility_Coordination'] = np.sign(data['Volume_Scaling'] - 1) * np.sign(data['Volatility_Scaling'] - 1)
    data['Fracture_Volume_Alignment'] = np.sign(data['Fracture_Acceleration']) * np.sign(data['Volume_Scaling'] - 1)
    
    # Gap-Efficiency Fracture
    prev_high_low_diff = data['high'].shift(1) - data['low'].shift(1)
    prev_high_low_diff = prev_high_low_diff.replace(0, np.nan)
    data['Overnight_Gap_Pressure'] = (data['open'] - data['close'].shift(1)) / prev_high_low_diff
    data['Intraday_Volatility_Efficiency'] = (data['close'] - data['open']) / data['TrueRange']
    data['Gap_Fracture_Asymmetry'] = data['Overnight_Gap_Pressure'] * data['Intraday_Volatility_Efficiency'] * data['Fracture_Acceleration']
    
    # Asymmetry Regime Fracture Detection
    # Volatility-Fracture Compression
    data['Price_Compression'] = data['TrueRange'] / data['TrueRange'].rolling(window=5).mean()
    data['Volume_Compression'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['Compression_Fracture_Asymmetry'] = data['Price_Compression'] * data['Volume_Compression'] * data['Range_Asymmetry_Fractality'] * data['Fracture_Acceleration']
    
    # Flow Asymmetry Fracture
    data['High_Flow_Fracture'] = (
        (data['Volume_Scaling'] > 1.2) & 
        (data['amount'] / data['amount'].rolling(window=5).mean() > 1.1) & 
        (data['Fracture_Acceleration'] > 1)
    )
    data['Low_Flow_Fracture'] = (
        (data['Volume_Scaling'] < 0.8) & 
        (data['amount'] / data['amount'].rolling(window=5).mean() < 0.9) & 
        (data['Fracture_Acceleration'] < 1)
    )
    data['Flow_Fracture_Score'] = data['High_Flow_Fracture'].astype(int) - data['Low_Flow_Fracture'].astype(int)
    
    # Microstructure Timing Fracture
    data['Opening_Liquidity_Pressure'] = (data['open'] - data['close'].shift(1)) / data['amount']
    data['Timing_Fracture_Momentum'] = (
        data['Opening_Liquidity_Pressure'] * 
        data['Intraday_Volatility_Efficiency'] * 
        data['Volume_Volatility_Coordination'] * 
        data['Fracture_Acceleration']
    )
    
    # Structural Break Fracture Patterns
    # Volume-Confirmed Fracture Breaks
    fracture_accel_abs = data['Fracture_Acceleration'].abs()
    data['Fracture_Break_Intensity'] = fracture_accel_abs / fracture_accel_abs.rolling(window=5).mean()
    data['Volume_Confirmation'] = data['Volume_Scaling']
    data['Break_Fracture_Signal'] = data['Fracture_Break_Intensity'] * data['Volume_Confirmation'] * data['Range_Asymmetry_Fractality']
    
    # Range Asymmetry Fracture Expansion
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    # Calculate rolling average of TrueRange for up and down days separately
    up_day_tr = data['TrueRange'].where(up_days)
    down_day_tr = data['TrueRange'].where(down_days)
    
    data['Up_Day_Range_Expansion'] = data['TrueRange'] / up_day_tr.rolling(window=5, min_periods=1).mean()
    data['Down_Day_Range_Compression'] = data['TrueRange'] / down_day_tr.rolling(window=5, min_periods=1).mean()
    data['Volatility_Fracture_Index'] = (data['Up_Day_Range_Expansion'] / data['Down_Day_Range_Compression']) * data['Fracture_Acceleration']
    
    # Fracture Asymmetry Persistence
    data['Consecutive_Fracture_Days'] = (data['Fracture_Acceleration'] > 1.1).rolling(window=5).sum()
    data['Volume_Fracture_Persistence'] = data['Fracture_Volume_Alignment'].rolling(window=5).mean()
    data['Break_Fracture_Persistence'] = data['Consecutive_Fracture_Days'] * data['Volume_Fracture_Persistence']
    
    # Trade Efficiency Fracture Asymmetry
    # Trade Size Fracture Concentration
    data['Average_Trade_Size'] = data['amount'] / data['volume']
    data['Size_Fracture_Momentum'] = data['Average_Trade_Size'] / data['Average_Trade_Size'].rolling(window=5).mean()
    data['Concentration_Fracture'] = data['Average_Trade_Size'] * data['Size_Fracture_Momentum'] * data['Range_Asymmetry_Fractality'] * data['Fracture_Acceleration']
    
    # Volume-Volatility Fracture Divergence
    data['Immediate_Volatility_Correlation'] = (data['close'] - data['open']) * data['volume'] * data['TrueRange']
    data['Lagged_Volatility_Correlation'] = (data['close'].shift(1) - data['open'].shift(1)) * data['volume'].shift(1) * data['TrueRange'].shift(1)
    data['Volatility_Fracture_Decay'] = (data['Immediate_Volatility_Correlation'] / data['Lagged_Volatility_Correlation']) * data['Fracture_Acceleration']
    
    # Microstructure Volatility Fracture Timing
    data['Amount_Volatility_Intensity'] = data['amount'] / data['TrueRange']
    data['Volatility_Fracture_Efficiency'] = data['Amount_Volatility_Intensity'] * data['Volume_Volatility_Coordination'] * data['Fracture_Acceleration']
    
    # Dynamic Fracture Asymmetry Weighting
    # Volatility Fracture Classification
    data['Short_term_Volatility_Fracture'] = data['Range_Asymmetry_Fractality'].rolling(window=5).mean()
    data['Medium_term_Volatility_Fracture'] = data['Range_Asymmetry_Fractality'].rolling(window=20).mean()
    data['Volatility_Fracture_Regime'] = (data['Short_term_Volatility_Fracture'] / data['Medium_term_Volatility_Fracture']) * data['Fracture_Acceleration']
    
    # Fracture Asymmetry Coherence
    fracture_regimes = [
        np.sign(data['Fracture_Acceleration']),
        np.sign(data['Volume_Scaling'] - 1),
        np.sign(data['Volatility_Scaling'] - 1)
    ]
    data['Fracture_Alignment'] = sum((regime1 == regime2).astype(int) for i, regime1 in enumerate(fracture_regimes) for regime2 in fracture_regimes[i+1:])
    
    # Simple persistence calculation
    data['Fracture_Persistence'] = 1
    for i in range(1, len(data)):
        if all(data.iloc[i][['Fracture_Acceleration', 'Volume_Scaling', 'Volatility_Scaling']].notna()):
            current_signs = [
                np.sign(data.iloc[i]['Fracture_Acceleration']),
                np.sign(data.iloc[i]['Volume_Scaling'] - 1),
                np.sign(data.iloc[i]['Volatility_Scaling'] - 1)
            ]
            prev_signs = [
                np.sign(data.iloc[i-1]['Fracture_Acceleration']),
                np.sign(data.iloc[i-1]['Volume_Scaling'] - 1),
                np.sign(data.iloc[i-1]['Volatility_Scaling'] - 1)
            ]
            if current_signs == prev_signs:
                data.iloc[i, data.columns.get_loc('Fracture_Persistence')] = data.iloc[i-1]['Fracture_Persistence'] + 1
    
    data['Fracture_Coherence_Score'] = data['Fracture_Alignment'] * data['Fracture_Persistence']
    
    # Flow Fracture Confirmation
    data['Volume_Fracture_Confirmation'] = data['Volume_Scaling']
    data['Volatility_Fracture_Confirmation'] = data['Volatility_Scaling']
    data['Flow_Fracture_Consistency'] = (
        np.sign(data['Volume_Fracture_Confirmation']) * 
        np.sign(data['Volatility_Fracture_Confirmation']) * 
        np.sign(data['Fracture_Acceleration'])
    )
    
    # Fracture Cluster & Asymmetry Framework
    # Cluster Detection System
    data['Cluster_Density'] = (data['Fracture_Acceleration'] > 0).rolling(window=3).sum()
    data['Cluster_Magnitude'] = data['Fracture_Acceleration'].abs().rolling(window=3).mean()
    data['Multi_Fracture_Coherence'] = (
        np.sign(data['Short_Term_Fracture']) * 
        np.sign(data['Medium_Term_Fracture']) * 
        np.sign(data['Fracture_Acceleration'])
    )
    data['Volume_Cluster_Intensity'] = data['Cluster_Magnitude'] * data['volume'].rolling(window=3).mean() * data['Range_Asymmetry_Fractality']
    
    # Multi-Dimensional Fracture Confirmation
    data['Fracture_Volume_Correlation'] = np.sign(data['Fracture_Acceleration']) * np.sign(data['Volume_Scaling'] - 1)
    data['Efficiency_Fracture_Alignment'] = np.sign(data['Intraday_Volatility_Efficiency']) * np.sign(data['Fracture_Acceleration'])
    data['Amount_Fracture_Direction'] = np.sign(data['amount'] / data['amount'].rolling(window=5).mean() - 1) * np.sign(data['Fracture_Acceleration'])
    data['Multi_Fracture_Confirmation_Score'] = data['Fracture_Volume_Correlation'] * data['Efficiency_Fracture_Alignment'] * data['Amount_Fracture_Direction']
    
    # Cluster-Asymmetry Integration
    data['Cluster_Quality'] = data['Cluster_Density'] * data['Multi_Fracture_Coherence']
    data['Asymmetry_Weighted_Cluster'] = data['Cluster_Quality'] * data['Multi_Fracture_Confirmation_Score'] * data['Range_Asymmetry_Fractality']
    data['Volume_Scaled_Cluster'] = data['Asymmetry_Weighted_Cluster'] / data['volume']
    
    # Adaptive Alpha Construction
    # Core Fracture Asymmetry Integration
    data['Fractality_Foundation'] = data['Range_Asymmetry_Fractality'] * data['Volume_Volatility_Coordination'] * data['Gap_Fracture_Asymmetry']
    data['Microstructure_Dynamics'] = data['Compression_Fracture_Asymmetry'] * data['Timing_Fracture_Momentum'] * data['Volatility_Fracture_Efficiency']
    data['Structural_Signals'] = data['Break_Fracture_Signal'] * data['Volatility_Fracture_Index'] * data['Break_Fracture_Persistence']
    data['Efficiency_Components'] = data['Concentration_Fracture'] * data['Volatility_Fracture_Decay'] * data['Volume_Scaled_Cluster']
    
    # Regime-Adaptive Fracture Weighting
    data['Volatility_Adjustment'] = (
        data['Fractality_Foundation'] * 
        data['Microstructure_Dynamics'] * 
        data['Structural_Signals'] * 
        data['Efficiency_Components'] * 
        data['Volatility_Fracture_Regime']
    )
    data['Flow_Fracture_Multiplier'] = data['Volatility_Adjustment'] * data['Flow_Fracture_Score']
    data['Coherence_Enhancement'] = data['Flow_Fracture_Multiplier'] * data['Fracture_Coherence_Score']
    
    # Final Alpha Construction
    data['Raw_Fracture_Alpha'] = data['Coherence_Enhancement']
    data['Flow_Confirmed_Fracture_Alpha'] = data['Raw_Fracture_Alpha'] * data['Flow_Fracture_Consistency']
    
    # Clean up and return
    alpha_series = data['Flow_Confirmed_Fracture_Alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_series
