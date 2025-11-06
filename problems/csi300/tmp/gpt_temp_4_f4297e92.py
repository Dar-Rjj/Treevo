import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Fractal Dynamics
    df['Intraday_Fracture'] = (df['high'] - df['low']) / df['close']
    df['Overnight_Fracture'] = abs(df['open'] / df['close'].shift(1) - 1)
    
    # Resonance Momentum
    df['Fast_Resonance'] = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) - df['close'].shift(2))
    df['Slow_Resonance'] = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) - df['close'].shift(6))
    
    # Pressure Dynamics
    df['Opening_Pressure'] = (df['open'] - (df['high'].shift(1) + df['low'].shift(1)) / 2) / ((df['high'].shift(1) - df['low'].shift(1)) / 2)
    df['Closing_Pressure'] = (df['close'] - (df['high'] + df['low']) / 2) / ((df['high'] - df['low']) / 2)
    
    # Volume Asymmetry
    df['Volume_Flow_Ratio'] = np.where(df['close'] > df['open'], df['volume'], np.where(df['close'] < df['open'], df['volume'], np.nan))
    df['Volume_Flow_Ratio'] = df.groupby(df.index)['Volume_Flow_Ratio'].transform(lambda x: x[df['close'] > df['open']].mean() / x[df['close'] < df['open']].mean() if len(x[df['close'] > df['open']]) > 0 and len(x[df['close'] < df['open']]) > 0 else 1)
    
    df['Volatility_Skew_Ratio'] = ((df['high'] - df['close']) / (df['close'] - df['low'])) / ((df['open'] - df['close'].shift(1)) / (df['high'] - df['low']))
    
    # Multi-Timeframe Patterns
    df['Momentum_Divergence'] = (df['close'] / df['close'].shift(5) - 1) - (df['close'] / df['close'].shift(20) - 1)
    df['Gap_Resonance'] = abs(df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1))
    
    # Core Components
    df['Fractal_Momentum'] = (df['Intraday_Fracture'] + df['Overnight_Fracture']) * df['Momentum_Divergence']
    df['Resonance_Coupling'] = df['Fast_Resonance'] * df['Slow_Resonance']
    df['Pressure_Dynamics'] = df['Opening_Pressure'] * df['Closing_Pressure']
    
    # Alpha Synthesis
    df['Base_Signal'] = df['Fractal_Momentum'] * df['Resonance_Coupling'] * df['Pressure_Dynamics']
    df['Enhanced_Signal'] = df['Base_Signal'] * df['Volume_Flow_Ratio'] * df['Volatility_Skew_Ratio']
    df['Fractal_Momentum_Resonance_Alpha'] = df['Enhanced_Signal'] * df['Gap_Resonance']
    
    # Handle infinite values and NaN values
    df['Fractal_Momentum_Resonance_Alpha'] = df['Fractal_Momentum_Resonance_Alpha'].replace([np.inf, -np.inf], np.nan)
    
    return df['Fractal_Momentum_Resonance_Alpha']
