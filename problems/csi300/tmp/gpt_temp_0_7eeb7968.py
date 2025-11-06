import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Adaptive Regime Price-Volume Momentum Divergence factor
    """
    df = data.copy()
    
    # Initialize all EMA columns
    df['EMA_fast_p'] = np.nan
    df['EMA_med_p'] = np.nan
    df['EMA_slow_p'] = np.nan
    df['EMA_fast_v'] = np.nan
    df['EMA_med_v'] = np.nan
    df['EMA_slow_v'] = np.nan
    df['EMA_range'] = np.nan
    df['EMA_vol_chg'] = np.nan
    
    # Calculate EMA components iteratively
    for i in range(len(df)):
        if i == 0:
            # Initialize with first values
            df.loc[df.index[i], 'EMA_fast_p'] = df['close'].iloc[i]
            df.loc[df.index[i], 'EMA_med_p'] = df['close'].iloc[i]
            df.loc[df.index[i], 'EMA_slow_p'] = df['close'].iloc[i]
            df.loc[df.index[i], 'EMA_fast_v'] = df['volume'].iloc[i]
            df.loc[df.index[i], 'EMA_med_v'] = df['volume'].iloc[i]
            df.loc[df.index[i], 'EMA_slow_v'] = df['volume'].iloc[i]
            
            # Range and volume change
            df.loc[df.index[i], 'Range'] = df['high'].iloc[i] - df['low'].iloc[i]
            df.loc[df.index[i], 'EMA_range'] = df['Range'].iloc[i]
            
            df.loc[df.index[i], 'Vol_Chg'] = 0
            df.loc[df.index[i], 'EMA_vol_chg'] = 0
        else:
            # Price EMAs
            df.loc[df.index[i], 'EMA_fast_p'] = 0.5 * df['close'].iloc[i] + 0.5 * df['EMA_fast_p'].iloc[i-1]
            df.loc[df.index[i], 'EMA_med_p'] = 0.2 * df['close'].iloc[i] + 0.8 * df['EMA_med_p'].iloc[i-1]
            df.loc[df.index[i], 'EMA_slow_p'] = 0.125 * df['close'].iloc[i] + 0.875 * df['EMA_slow_p'].iloc[i-1]
            
            # Volume EMAs
            df.loc[df.index[i], 'EMA_fast_v'] = 0.5 * df['volume'].iloc[i] + 0.5 * df['EMA_fast_v'].iloc[i-1]
            df.loc[df.index[i], 'EMA_med_v'] = 0.2 * df['volume'].iloc[i] + 0.8 * df['EMA_med_v'].iloc[i-1]
            df.loc[df.index[i], 'EMA_slow_v'] = 0.125 * df['volume'].iloc[i] + 0.875 * df['EMA_slow_v'].iloc[i-1]
            
            # Range and EMA range
            df.loc[df.index[i], 'Range'] = df['high'].iloc[i] - df['low'].iloc[i]
            df.loc[df.index[i], 'EMA_range'] = 0.2 * df['Range'].iloc[i] + 0.8 * df['EMA_range'].iloc[i-1]
            
            # Volume change and EMA volume change
            df.loc[df.index[i], 'Vol_Chg'] = abs(df['volume'].iloc[i] - df['volume'].iloc[i-1])
            df.loc[df.index[i], 'EMA_vol_chg'] = 0.2 * df['Vol_Chg'].iloc[i] + 0.8 * df['EMA_vol_chg'].iloc[i-1]
    
    # Volatility scales
    df['Price_Vol_Scale'] = 1 / (df['EMA_range'] + 0.0001)
    df['Volume_Vol_Scale'] = 1 / (df['EMA_vol_chg'] + 0.0001)
    
    # Price momentum ratios (scaled)
    df['Price_Fast_Med'] = (df['EMA_fast_p'] / df['EMA_med_p'] - 1) * df['Price_Vol_Scale']
    df['Price_Med_Slow'] = (df['EMA_med_p'] / df['EMA_slow_p'] - 1) * df['Price_Vol_Scale']
    df['Price_Fast_Slow'] = (df['EMA_fast_p'] / df['EMA_slow_p'] - 1) * df['Price_Vol_Scale']
    
    # Volume momentum ratios (scaled)
    df['Volume_Fast_Med'] = (df['EMA_fast_v'] / df['EMA_med_v'] - 1) * df['Volume_Vol_Scale']
    df['Volume_Med_Slow'] = (df['EMA_med_v'] / df['EMA_slow_v'] - 1) * df['Volume_Vol_Scale']
    df['Volume_Fast_Slow'] = (df['EMA_fast_v'] / df['EMA_slow_v'] - 1) * df['Volume_Vol_Scale']
    
    # Regime transition weights
    df['Volatility_Ratio'] = df['EMA_range'] / (df['EMA_range'].shift(5).fillna(df['EMA_range']) + 0.0001)
    df['Fast_Regime_Weight'] = np.maximum(0, np.minimum(1, 2.5 - df['Volatility_Ratio']))
    df['Medium_Regime_Weight'] = np.maximum(0, np.minimum(1, 1.5 - abs(1.2 - df['Volatility_Ratio'])))
    df['Slow_Regime_Weight'] = np.maximum(0, np.minimum(1, df['Volatility_Ratio'] - 0.8))
    
    # Momentum divergence construction
    df['Fast_Divergence'] = df['Price_Fast_Med'] - df['Volume_Fast_Med']
    df['Medium_Divergence'] = df['Price_Med_Slow'] - df['Volume_Med_Slow']
    df['Slow_Divergence'] = df['Price_Fast_Slow'] - df['Volume_Fast_Slow']
    
    # Bounded factor components
    df['Fast_Component'] = np.tanh(df['Fast_Divergence']) * df['Fast_Regime_Weight']
    df['Medium_Component'] = np.tanh(df['Medium_Divergence']) * df['Medium_Regime_Weight']
    df['Slow_Component'] = np.tanh(df['Slow_Divergence']) * df['Slow_Regime_Weight']
    
    # Final alpha output
    df['alpha'] = df['Fast_Component'] + df['Medium_Component'] + df['Slow_Component']
    
    return df['alpha']
