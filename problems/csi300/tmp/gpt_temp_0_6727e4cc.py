import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Momentum Regime Classification
    # Intraday Momentum Signature
    data['Opening_Momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['Closing_Momentum'] = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'].shift(1) + epsilon)
    
    # Multi-Day Momentum Persistence
    data['Close_2d_max'] = data['close'].rolling(window=3, min_periods=1).max()
    data['Close_2d_min'] = data['close'].rolling(window=3, min_periods=1).min()
    data['Close_5d_max'] = data['close'].shift(3).rolling(window=3, min_periods=1).max()
    data['Close_5d_min'] = data['close'].shift(3).rolling(window=3, min_periods=1).min()
    
    data['Momentum_Range_Expansion'] = (data['Close_2d_max'] - data['Close_2d_min']) / (data['Close_5d_max'] - data['Close_5d_min'] + epsilon)
    
    # Calculate Momentum Persistence Score
    momentum_expansion_binary = (data['Momentum_Range_Expansion'] > 1).astype(int)
    data['Momentum_Persistence_Count'] = momentum_expansion_binary.rolling(window=6, min_periods=1).sum()
    data['Momentum_Persistence_Score'] = data['Momentum_Persistence_Count'] * data['Momentum_Range_Expansion']
    
    # Price Divergence Framework
    data['Price_Range'] = data['high'] - data['low']
    data['Daily_Divergence'] = (data['close'] - data['open']) / (data['Price_Range'] + epsilon)
    data['Gap_Divergence'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + epsilon)
    
    # Divergence Momentum
    data['Price_Divergence_Change'] = data['Daily_Divergence'] - data['Daily_Divergence'].shift(1)
    
    # Calculate Divergence Persistence
    daily_div_sign = np.sign(data['Daily_Divergence'])
    same_sign_count = daily_div_sign.rolling(window=4, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    data['Divergence_Persistence'] = same_sign_count * data['Daily_Divergence']
    
    # Volume Divergence Framework
    data['Volume_Divergence'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    
    # Morning and Afternoon Volume Divergence (simplified as same calculation for now)
    volume_div_base = data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    data['Morning_Volume_Divergence'] = volume_div_base
    data['Afternoon_Volume_Divergence'] = volume_div_base
    
    # Volume Divergence Dynamics
    data['Volume_Distribution_Divergence'] = data['Morning_Volume_Divergence'] - data['Afternoon_Volume_Divergence']
    
    data['Value_Volume_Divergence'] = (data['amount'] / (data['amount'].shift(1) + epsilon)) / (data['volume'] / (data['volume'].shift(1) + epsilon))
    data['Volume_Divergence_Change'] = data['Volume_Divergence'] - data['Volume_Divergence'].shift(1)
    
    # Regime-Adaptive Core Signals
    # High Momentum Regime Signals
    data['High_Mom_Price_Signal'] = data['Daily_Divergence'] * data['Gap_Divergence'] * data['Volume_Divergence']
    
    # Calculate count of same sign for Volume Distribution Divergence
    vdd_sign = np.sign(data['Volume_Distribution_Divergence'])
    vdd_same_sign_count = vdd_sign.rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    data['High_Mom_Volume_Signal'] = data['Volume_Distribution_Divergence'] * vdd_same_sign_count
    
    # Low Momentum Regime Signals
    data['Low_Mom_Price_Signal'] = (data['Daily_Divergence'] / (data['Daily_Divergence'].shift(1) + epsilon)) * data['Value_Volume_Divergence']
    data['Low_Mom_Volume_Signal'] = data['Volume_Distribution_Divergence'] * data['Divergence_Persistence']
    
    # Microstructure Quality Assessment
    # Pattern Divergence
    opening_closing_diff = data['Opening_Momentum'] - data['Closing_Momentum']
    op_cl_sign = np.sign(opening_closing_diff)
    data['Regime_Divergence'] = op_cl_sign.rolling(window=6, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    dd_sign = np.sign(data['Daily_Divergence'])
    data['Divergence_Consistency'] = dd_sign.rolling(window=6, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    # Volume Quality
    data['Volume_Flow_Divergence'] = vdd_same_sign_count
    
    vd_dd_alignment = np.sign(data['Volume_Divergence'] * data['Daily_Divergence'])
    data['Volume_Divergence_Alignment'] = vd_dd_alignment.rolling(window=6, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    
    # Multi-Timeframe Integration
    # Short-term Integration
    data['Intraday_Divergence_Momentum'] = data['Price_Divergence_Change'] * data['Volume_Divergence_Change']
    data['Volume_Regime_Divergence'] = np.sign(data['Volume_Distribution_Divergence']) * np.sign(data['Opening_Momentum'] - data['Closing_Momentum'])
    
    # Medium-term Integration
    vdd_sign_4d = np.sign(data['Volume_Distribution_Divergence'])
    vdd_same_sign_count_4d = vdd_sign_4d.rolling(window=5, min_periods=1).apply(
        lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 1, raw=False
    )
    data['Multi_Day_Divergence_Persistence'] = data['Divergence_Persistence'] * vdd_same_sign_count_4d
    data['Regime_Divergence_Alignment'] = np.sign(data['Daily_Divergence']) * np.sign(data['Momentum_Persistence_Score'])
    
    # Final Alpha Construction
    # Core Signal Components
    data['Regime_Adaptive_Divergence'] = np.where(
        data['Momentum_Range_Expansion'] > 1,
        data['High_Mom_Price_Signal'] * data['High_Mom_Volume_Signal'],
        data['Low_Mom_Price_Signal'] * data['Low_Mom_Volume_Signal']
    )
    
    data['Volume_Divergence_Component'] = data['Volume_Distribution_Divergence'] * (data['Volume_Flow_Divergence'] * data['Volume_Divergence_Alignment'])
    data['Quality_Enhancement'] = (data['Regime_Divergence'] * data['Divergence_Consistency']) * (data['Volume_Flow_Divergence'] * data['Volume_Divergence_Alignment'])
    
    # Signal Integration
    data['Divergence_Volume_Synthesis'] = data['Regime_Adaptive_Divergence'] * data['Volume_Divergence_Component']
    data['Quality_Regime_Enhancement'] = data['Divergence_Volume_Synthesis'] * data['Quality_Enhancement']
    
    # Multi-Timeframe Integration
    data['Multi_Timeframe_Integration'] = (data['Intraday_Divergence_Momentum'] * data['Volume_Regime_Divergence']) * (data['Multi_Day_Divergence_Persistence'] * data['Regime_Divergence_Alignment'])
    
    # Final Alpha
    alpha = data['Quality_Regime_Enhancement'] * data['Multi_Timeframe_Integration']
    
    return alpha
