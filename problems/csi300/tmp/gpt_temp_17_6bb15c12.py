import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate TrueRange
    df['TrueRange'] = np.maximum(df['high'] - df['low'], 
                                np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                          abs(df['low'] - df['close'].shift(1))))
    
    # Asymmetric Gap-Range Dynamics
    df['Upside_Gap_Acceleration'] = (df['open'] / df['close'].shift(1) - 1) - (df['open'].shift(1) / df['close'].shift(2) - 1)
    df['Downside_Gap_Acceleration'] = (df['low'] - df['low'].shift(1)) - (df['low'].shift(1) - df['low'].shift(2))
    df['Gap_Range_Asymmetry_Ratio'] = df['Upside_Gap_Acceleration'] / (abs(df['Downside_Gap_Acceleration']) + 0.001)
    
    # Multi-Timeframe Gap-Range Volume Divergence
    # Short-term (5-day) Divergence
    df['Gap_Range_Momentum_5'] = (df['open'] / df['close'].shift(1) - 1) - (df['open'].shift(5) / df['close'].shift(6) - 1)
    df['Volume_Momentum_5'] = (df['volume'] - df['volume'].shift(5)) / (df['volume'].shift(5) + 1e-8)
    df['Short_Divergence'] = np.sign(df['Gap_Range_Momentum_5']) * np.sign(df['Volume_Momentum_5']) * (abs(df['Gap_Range_Momentum_5']) - abs(df['Volume_Momentum_5']))
    
    # Medium-term (10-day) Divergence
    df['Gap_Range_Momentum_10'] = (df['open'] / df['close'].shift(1) - 1) - (df['open'].shift(10) / df['close'].shift(11) - 1)
    df['Volume_Momentum_10'] = (df['volume'] - df['volume'].shift(10)) / (df['volume'].shift(10) + 1e-8)
    df['Medium_Divergence'] = np.sign(df['Gap_Range_Momentum_10']) * np.sign(df['Volume_Momentum_10']) * (abs(df['Gap_Range_Momentum_10']) - abs(df['Volume_Momentum_10']))
    
    # Divergence Consistency
    df['Divergence_Consistency'] = np.sign(df['Short_Divergence']) * np.sign(df['Medium_Divergence']) * np.minimum(abs(df['Short_Divergence']), abs(df['Medium_Divergence']))
    
    # Fractal Gap-Range Volume Structure
    gap_range_changes = []
    for i in range(len(df)):
        if i >= 10:
            numerator = abs((df['open'].iloc[i] / df['close'].shift(1).iloc[i] - 1) - (df['open'].shift(10).iloc[i] / df['close'].shift(11).iloc[i] - 1))
            denominator = 0
            for j in range(10):
                if i - j >= 1:
                    change = abs((df['open'].iloc[i-j] / df['close'].shift(1).iloc[i-j] - 1) - (df['open'].shift(1).iloc[i-j] / df['close'].shift(2).iloc[i-j] - 1))
                    denominator += change
            gap_range_changes.append(numerator / (denominator + 1e-8) if denominator != 0 else 0)
        else:
            gap_range_changes.append(0)
    
    df['Gap_Range_Fractal_Efficiency'] = gap_range_changes
    df['Asymmetric_Gap_Range_Fractal_Momentum'] = df['Gap_Range_Fractal_Efficiency'] * df['Gap_Range_Asymmetry_Ratio']
    
    # Volume-Weighted Gap Acceleration
    df['Volume_Persistence'] = df['volume'] / (df['volume'].shift(5) + 1e-8)
    df['Net_Gap_Volume_Bias'] = (df['Upside_Gap_Acceleration'] * df['Volume_Persistence']) - (df['Downside_Gap_Acceleration'] * (df['volume'] / (df['volume'].shift(1) + 1e-8) - 1))
    
    # Gap-Range Flow Efficiency
    df['Daily_Gap_Range_Flow'] = df['amount'] * (df['close'] - df['open']) / (df['TrueRange'] + 1e-8)
    
    # Calculate rolling high-low ranges
    df['High_Low_Range_2'] = df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()
    df['High_Low_Range_4'] = df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()
    
    df['Short_term_Gap_Efficiency'] = (df['open'] - df['close'].shift(1)) / (df['High_Low_Range_2'] + 1e-8)
    df['Medium_term_Gap_Efficiency'] = (df['open'] - df['close'].shift(1)) / (df['High_Low_Range_4'] + 1e-8)
    
    # Volatility-Regime Adaptive Integration
    df['TrueRange_Increase_Count'] = (df['TrueRange'] > df['TrueRange'].shift(1)).rolling(window=5).sum()
    df['Volatility_Context'] = df['TrueRange'] / (df['high'] - df['low'] + 1e-8) + df['TrueRange_Increase_Count']
    
    df['Gap_Efficiency_Divergence'] = df['Short_term_Gap_Efficiency'] - df['Medium_term_Gap_Efficiency']
    
    # Regime-Adaptive Output
    conditions = [
        (df['Gap_Range_Fractal_Efficiency'] > 0.6) & (df['Divergence_Consistency'] > 0),
        (df['Gap_Range_Fractal_Efficiency'] < 0.4) & (abs(df['Short_Divergence'] - df['Medium_Divergence']) > 0)
    ]
    choices = [1.3, 0.8]
    df['Regime_Multiplier'] = np.select(conditions, choices, default=(df['volume'] / (df['volume'].shift(5) + 1e-8) - df['volume'] / (df['volume'].shift(10) + 1e-8)))
    
    df['Regime_Adaptive_Output'] = (df['Asymmetric_Gap_Range_Fractal_Momentum'] * df['Net_Gap_Volume_Bias'] * 
                                   (df['Short_Divergence'] + df['Medium_Divergence'])) * df['Regime_Multiplier']
    
    # Gap Pattern Integration
    df['Gap_Direction'] = np.sign(df['open'] - df['close'].shift(1))
    df['Intraday_Direction'] = np.sign(df['close'] - df['open'])
    df['Gap_Pattern_Alignment'] = df['Gap_Direction'] * df['Intraday_Direction']
    
    # Final Alpha Factor
    df['TrueRange_4_avg'] = df['TrueRange'].rolling(window=5).mean()
    df['TrueRange_12_avg'] = df['TrueRange'].rolling(window=13).mean()
    df['Gap_Expansion_Confirmation'] = (df['TrueRange'] / (df['TrueRange_4_avg'] + 1e-8)) * (df['TrueRange'] / (df['TrueRange_12_avg'] + 1e-8))
    
    df['Gap_Range_Volume_Momentum_Alpha'] = (df['Regime_Adaptive_Output'] * df['Gap_Expansion_Confirmation'] * 
                                            df['Gap_Pattern_Alignment'])
    
    # Quality Adjustment
    df['Quality_Adjustment'] = 1 - 0.2 * abs(df['Gap_Pattern_Alignment'] - 1)
    df['Final_Alpha'] = df['Gap_Range_Volume_Momentum_Alpha'] * df['Quality_Adjustment']
    
    return df['Final_Alpha']
