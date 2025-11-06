import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['TrueRange'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Multi-Scale Gap Efficiency
    df['ShortTerm_GapEfficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['MediumTerm_GapEfficiency'] = abs(df['close'] - df['open'].shift(5)) / df['high'].rolling(6).apply(lambda x: (x - df.loc[x.index, 'low']).sum(), raw=False)
    df['GapEfficiency_Divergence'] = df['ShortTerm_GapEfficiency'] - df['MediumTerm_GapEfficiency']
    
    # Flow-Momentum Divergence
    df['Momentum_Difference'] = (df['close'] / df['close'].shift(4) - 1) - (df['close'] / df['close'].shift(9) - 1)
    df['Amount_Flow_Ratio'] = df['amount'].rolling(5).mean() / df['amount'].rolling(15).mean() - 1
    df['FlowMomentum_Divergence'] = df['Momentum_Difference'] * df['Amount_Flow_Ratio']
    
    # Combined Gap-Flow Divergence
    df['GapFlow_Divergence'] = np.sqrt(df['GapEfficiency_Divergence'] * df['FlowMomentum_Divergence'])
    
    # Intraday Pressure Asymmetry
    df['Morning_Gap_Pressure'] = (df['high'] - df['open']) / abs(df['open'] - df['close'].shift(1))
    df['Gap_Fill_Efficiency'] = (df['close'] - df['open']) / abs(df['open'] - df['close'].shift(1))
    df['Pressure_Asymmetry'] = df['Morning_Gap_Pressure'] - df['Gap_Fill_Efficiency']
    
    # Volume Efficiency Asymmetry
    returns = df['close'].pct_change()
    up_days = returns > 0
    df['Upside_Volume_Ratio'] = df['volume'].rolling(10).apply(lambda x: x[up_days.loc[x.index]].mean() / x.mean(), raw=False)
    
    pos_returns = np.maximum(returns, 0)
    neg_returns = np.maximum(-returns, 0)
    df['Price_Efficiency_Asymmetry'] = np.log(1 + pos_returns.rolling(10).sum()) - np.log(1 + neg_returns.rolling(10).sum())
    df['Volume_Efficiency_Asymmetry'] = df['Upside_Volume_Ratio'] * df['Price_Efficiency_Asymmetry']
    
    # Combined Pressure Efficiency
    df['Pressure_Efficiency'] = np.cbrt(df['Pressure_Asymmetry'] * df['Volume_Efficiency_Asymmetry'])
    
    # Range Compression with Flow Efficiency
    df['Range_Compression'] = (df['high'] - df['low']).rolling(5).sum() / (df['high'] - df['low']).rolling(10).sum() - 1
    df['Flow_Efficiency'] = abs(df['close'] - df['close'].shift(1)) / df['amount']
    df['RangeFlow_Combined'] = df['Range_Compression'] * df['Flow_Efficiency']
    
    # Breakout Detection with Volume Coherence
    df['PriceVolume_Correlation'] = df['close'].rolling(3).corr(df['volume'])
    df['Movement_Efficiency'] = abs(df['close'] - df['close'].shift(1)) / df['TrueRange']
    df['Breakout_Detection'] = df['PriceVolume_Correlation'] * df['Movement_Efficiency']
    
    # Efficiency Breakout Signal
    df['Efficiency_Breakout'] = df['RangeFlow_Combined'] * df['Breakout_Detection']
    
    # Flow Fractal Analysis
    df['ShortTerm_Flow'] = df['amount'] / df['amount'].rolling(5).mean()
    df['MediumTerm_Flow'] = df['amount'].rolling(5).mean() / df['amount'].rolling(15).mean()
    df['Flow_Fractal_Divergence'] = df['ShortTerm_Flow'] - df['MediumTerm_Flow']
    
    # Volume Cluster Efficiency
    df['Gap_Volume_Fractal'] = np.log(df['volume'].rolling(8).max() - df['volume'].rolling(8).min()) / np.log(df['volume'].rolling(3).max() - df['volume'].rolling(3).min())
    
    df['Flow_Turnover_Momentum'] = df['amount'] / df['amount'].rolling(4).apply(lambda x: x.iloc[:-1].max(), raw=False)
    
    amount_median = df['amount'].rolling(8).median()
    df['Cluster_Duration'] = df['amount'].rolling(8).apply(lambda x: (x > 2.5 * amount_median.loc[x.index]).sum(), raw=False)
    
    df['Volume_Cluster_Efficiency'] = df['Gap_Volume_Fractal'] * df['Flow_Turnover_Momentum'] * df['Cluster_Duration']
    
    # Combined Flow Dynamics
    df['Combined_Flow_Dynamics'] = df['Flow_Fractal_Divergence'] * df['Volume_Cluster_Efficiency']
    
    # Final Alpha Synthesis
    df['Base_Signal'] = df['GapFlow_Divergence'] * df['Pressure_Efficiency']
    df['Efficiency_Multiplier'] = df['Base_Signal'] * df['Efficiency_Breakout']
    df['Flow_Dynamics_Adjustment'] = df['Efficiency_Multiplier'] * df['Combined_Flow_Dynamics']
    
    # Volatility Scaling
    df['Volatility_Scaling'] = df['TrueRange'] / df['TrueRange'].rolling(20).mean()
    df['Final_Alpha'] = df['Flow_Dynamics_Adjustment'] * df['Volatility_Scaling']
    
    return df['Final_Alpha']
