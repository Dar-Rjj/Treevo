import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Intraday Price Fractal Structure
    Morning_Impulse_Strength = (df['high'] - df['open']) * (df['close'] - df['open'])
    Afternoon_Impulse_Strength = (df['open'] - df['low']) * (df['open'] - df['close'])
    Intraday_Fractal_Asymmetry = Morning_Impulse_Strength - Afternoon_Impulse_Strength
    Price_Fractal_Quality = (df['close'] - df['open']) * (df['high'] - df['low']) / (np.abs(df['close'] - df['open']) + 0.001)
    
    # Multi-Timeframe Volume Fractal
    Volume_Impulse_Short = df['volume'] * (df['close'] - df['open']) / (df['high'] - df['low'] + 0.001)
    
    # Rolling calculations for medium and long term
    Volume_Impulse_Medium = pd.Series(index=df.index, dtype=float)
    Volume_Impulse_Long = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 3:
            high_max_3 = df['high'].iloc[i-3:i+1].max()
            low_min_3 = df['low'].iloc[i-3:i+1].min()
            Volume_Impulse_Medium.iloc[i] = df['volume'].iloc[i] * (df['close'].iloc[i] - df['close'].iloc[i-3]) / (high_max_3 - low_min_3 + 0.001)
        else:
            Volume_Impulse_Medium.iloc[i] = 0
            
        if i >= 8:
            high_max_8 = df['high'].iloc[i-8:i+1].max()
            low_min_8 = df['low'].iloc[i-8:i+1].min()
            Volume_Impulse_Long.iloc[i] = df['volume'].iloc[i] * (df['close'].iloc[i] - df['close'].iloc[i-8]) / (high_max_8 - low_min_8 + 0.001)
        else:
            Volume_Impulse_Long.iloc[i] = 0
    
    Volume_Fractal_Divergence = Volume_Impulse_Short * Volume_Impulse_Medium * Volume_Impulse_Long
    
    # Price-Volume Fractal Synchronization
    Short_Term_Sync = Price_Fractal_Quality * Volume_Impulse_Short * np.sign(Intraday_Fractal_Asymmetry)
    
    Medium_Term_Sync = pd.Series(index=df.index, dtype=float)
    Long_Term_Sync = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 3:
            high_max_3 = df['high'].iloc[i-3:i+1].max()
            low_min_3 = df['low'].iloc[i-3:i+1].min()
            Medium_Term_Sync.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-3]) * Volume_Impulse_Medium.iloc[i] / (high_max_3 - low_min_3 + 0.001)
        else:
            Medium_Term_Sync.iloc[i] = 0
            
        if i >= 8:
            high_max_8 = df['high'].iloc[i-8:i+1].max()
            low_min_8 = df['low'].iloc[i-8:i+1].min()
            Long_Term_Sync.iloc[i] = (df['close'].iloc[i] - df['close'].iloc[i-8]) * Volume_Impulse_Long.iloc[i] / (high_max_8 - low_min_8 + 0.001)
        else:
            Long_Term_Sync.iloc[i] = 0
    
    Multi_Scale_Sync_Alignment = np.sign(Short_Term_Sync) * np.sign(Medium_Term_Sync) * np.sign(Long_Term_Sync)
    
    # Fractal Momentum Divergence Patterns
    Morning_Afternoon_Divergence = (Morning_Impulse_Strength / (Afternoon_Impulse_Strength + 0.001)) * (df['close'] - df['open'])
    Volume_Price_Divergence = Volume_Fractal_Divergence * Price_Fractal_Quality * np.sign(Intraday_Fractal_Asymmetry)
    Timeframe_Divergence_Cascade = (Volume_Impulse_Short / (Volume_Impulse_Medium + 0.001)) * (Volume_Impulse_Medium / (Volume_Impulse_Long + 0.001))
    Fractal_Divergence_Quality = Morning_Afternoon_Divergence * Volume_Price_Divergence * Timeframe_Divergence_Cascade
    
    # Multi-Scale Integration Core
    Short_Term_Core = Short_Term_Sync * Morning_Afternoon_Divergence
    Medium_Term_Core = Medium_Term_Sync * Volume_Price_Divergence
    Long_Term_Core = Long_Term_Sync * Timeframe_Divergence_Cascade
    Multi_Scale_Core_Alignment = np.sign(Short_Term_Core) * np.sign(Medium_Term_Core) * np.sign(Long_Term_Core)
    
    # Alpha Construction Framework
    Base_Fractal_Component = Intraday_Fractal_Asymmetry * Price_Fractal_Quality * Volume_Fractal_Divergence
    Divergence_Enhanced_Core = Base_Fractal_Component * Fractal_Divergence_Quality * Multi_Scale_Sync_Alignment
    Multi_Scale_Integrated = Divergence_Enhanced_Core * Multi_Scale_Core_Alignment * Short_Term_Core * Medium_Term_Core * Long_Term_Core
    
    # Momentum adjustment
    prev_close = df['close'].shift(1)
    Momentum_Adjusted_Alpha = Multi_Scale_Integrated * (df['close'] - df['open']) * (df['close'] - prev_close)
    
    # Final Alpha Factor
    Multi_Scale_Price_Volume_Fractal_Divergence_Alpha = Momentum_Adjusted_Alpha * np.sign(Intraday_Fractal_Asymmetry)
    
    return Multi_Scale_Price_Volume_Fractal_Divergence_Alpha
