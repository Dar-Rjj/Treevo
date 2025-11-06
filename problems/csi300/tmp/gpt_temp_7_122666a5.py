import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Fractal Gap Efficiency Analysis
    df['Micro_Gap_Efficiency'] = np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
    
    df['Macro_Gap_Efficiency'] = np.abs(df['close'] - df['open'].shift(5)) / (
        (df['high'] - df['low']).rolling(window=6).sum() + 0.0001
    )
    
    gap_diff = df['Micro_Gap_Efficiency'] - df['Macro_Gap_Efficiency']
    df['Fractal_Gap_Divergence'] = np.sign(gap_diff) * np.sqrt(np.abs(gap_diff))
    
    # Volume-Pressure Shadow Dynamics
    df['Price_Pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
    
    df['Volume_Pressure'] = df['volume'] / (
        df['volume'] + df['volume'].shift(1) + df['volume'].shift(2) + 0.0001
    )
    
    max_oc = np.maximum(df['open'], df['close'])
    min_oc = np.minimum(df['open'], df['close'])
    
    df['Upper_Shadow_Pressure'] = (df['high'] - max_oc) / (df['high'] - df['low'] + 0.0001)
    df['Lower_Shadow_Pressure'] = (min_oc - df['low']) / (df['high'] - df['low'] + 0.0001)
    
    df['Shadow_Asymmetry'] = (df['Upper_Shadow_Pressure'] - df['Lower_Shadow_Pressure']) * df['Volume_Pressure']
    
    # Amount Flow Momentum System
    df['Amount_Efficiency'] = (df['close'] - df['close'].shift(1)) * df['volume'] / (df['amount'] + 0.0001)
    
    vol_close = df['volume'] * df['close']
    df['Gap_Turnover_Momentum'] = vol_close / (
        vol_close.rolling(window=4).max().shift(1) + 0.0001
    )
    
    df['Nonlinear_Flow_Momentum'] = df['Amount_Efficiency'] * df['Gap_Turnover_Momentum'] * np.abs(df['Price_Pressure'])
    
    # Volatility Regime Detection
    df['Volatility_Expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 0.0001) > 1.2
    
    df['Volume_Surge'] = df['volume'] / df['volume'].shift(1) > 1.1
    
    gap_abs_4 = np.abs(df['close'] - df['open']).rolling(window=5).sum()
    gap_abs_9 = np.abs(df['close'] - df['open']).rolling(window=10).sum()
    df['Gap_Volatility_Compression'] = gap_abs_4 / (gap_abs_9 + 0.0001) - 1
    
    df['Regime_Amplifier'] = 1.0 + 0.4 * df['Volatility_Expansion'] + 0.3 * df['Volume_Surge'] + 0.3 * (1 - np.exp(-np.abs(df['Gap_Volatility_Compression'])))
    
    # Microstructure Consistency Confirmation
    def rolling_corr(x, y, window):
        return pd.Series([x.iloc[i-window+1:i+1].corr(y.iloc[i-window+1:i+1]) 
                         if i >= window-1 else np.nan for i in range(len(x))], index=x.index)
    
    df['Price_Volume_Correlation'] = rolling_corr(df['close'].diff(), df['volume'], 3)
    df['Shadow_Consistency'] = rolling_corr(df['Upper_Shadow_Pressure'], df['Lower_Shadow_Pressure'], 3)
    
    df['Microstructure_Score'] = df['Price_Volume_Correlation'] * (1 - np.abs(df['Shadow_Consistency']))
    
    # Alpha Synthesis
    core_signal1 = df['Fractal_Gap_Divergence'] * df['Shadow_Asymmetry']
    core_signal2 = df['Nonlinear_Flow_Momentum'] * df['Microstructure_Score']
    core_signal3 = df['Amount_Efficiency'] * df['Price_Pressure']
    
    base_alpha = (core_signal1 + core_signal2 + core_signal3) * df['Regime_Amplifier']
    
    final_alpha = base_alpha * np.sign(df['Fractal_Gap_Divergence']) * (1 + np.abs(df['Gap_Volatility_Compression']))
    
    return final_alpha
