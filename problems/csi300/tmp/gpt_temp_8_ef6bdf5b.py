import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Microstructure Momentum Synthesis alpha factor
    """
    # Calculate Upper and Lower Rejection
    df['Upper_Rejection'] = df['high'] - np.maximum(df['close'], df['open'])
    df['Lower_Rejection'] = np.minimum(df['close'], df['open']) - df['low']
    
    # Multi-Scale Rejection Dynamics
    # Asymmetric Rejection Momentum
    df['Directional_Rejection_Pressure'] = (df['Upper_Rejection'] - df['Lower_Rejection']) * np.sign(df['close'] - df['close'].shift(1))
    df['Volume_Confirmed_Rejection'] = (df['Upper_Rejection'] - df['Lower_Rejection']) * df['volume'] / df['volume'].shift(1)
    
    # Fractal Rejection Patterns
    df['Short_Term_Rejection_Momentum'] = (
        (df['high'] - df['close'].rolling(window=5).max()) - 
        (df['close'].rolling(window=5).min() - df['low'])
    ) / (df['high'] - df['low'] + 1e-8)
    
    df['Medium_Term_Rejection_Acceleration'] = (
        (df['high'] - df['close'].rolling(window=15).max()) - 
        (df['close'].rolling(window=15).min() - df['low'])
    ) / (df['high'] - df['low'] + 1e-8) - df['Short_Term_Rejection_Momentum']
    
    # Regime-Transition Microstructure
    # Volatility Regime Rejection
    df['High_Volatility_Rejection'] = (
        (df['Upper_Rejection'] - df['Lower_Rejection']) / (df['high'] - df['low'] + 1e-8) * 
        (df['high'] - df['low']) / (df['high'].shift(5) - df['low'].shift(5) + 1e-8)
    )
    
    # Volatility-Rejection Persistence
    def calc_persistence(series):
        if len(series) < 6:
            return np.nan
        signs = np.sign(series - series.shift(1))
        return (signs.rolling(window=5).apply(lambda x: (x == x.shift(1)).sum(), raw=False) / 5).iloc[-1]
    
    rejection_diff = df['Upper_Rejection'] - df['Lower_Rejection']
    df['Volatility_Rejection_Persistence'] = rejection_diff.rolling(window=6).apply(calc_persistence, raw=False)
    
    # Volume Regime Impact
    df['Volume_Surge_Rejection'] = (
        (df['Upper_Rejection'] - df['Lower_Rejection']) * 
        df['volume'] / ((df['volume'].shift(4) + df['volume'].shift(3) + df['volume'].shift(2) + df['volume'].shift(1)) / 4 + 1e-8)
    )
    
    df['Volume_Rejection_Regime'] = (
        np.sign(df['volume'] - df['volume'].shift(1)) * 
        np.sign(df['Upper_Rejection'] - df['Lower_Rejection'])
    )
    
    # Fractal Efficiency Dynamics
    # Gap Absorption Fractal
    df['Fractal_Gap_Momentum'] = (
        (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) - df['close'].shift(2) + 1e-8) * 
        df['volume'] / df['volume'].shift(1)
    )
    
    df['Gap_Rejection_Efficiency'] = (
        abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8) * 
        (df['Upper_Rejection'] - df['Lower_Rejection'])
    )
    
    # Intraday Efficiency Patterns
    df['Opening_Efficiency_Momentum'] = (
        (df['high'] - df['open']) - (df['open'] - df['low']) * 
        abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    )
    
    df['Closing_Efficiency_Fractal'] = (
        (df['close'] - (df['high'] + df['low']) / 2) / (df['high'] - df['low'] + 1e-8) * 
        df['volume'] / df['volume'].shift(1)
    )
    
    # Nonlinear Divergence Systems
    # Rejection-Efficiency Divergence
    current_rej_eff = (df['Upper_Rejection'] - df['Lower_Rejection']) / (df['high'] - df['low'] + 1e-8)
    prev_rej_eff = (df['Upper_Rejection'].shift(1) - df['Lower_Rejection'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    current_efficiency = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    prev_efficiency = abs(df['close'].shift(1) - df['open'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    
    df['Bullish_Microstructure_Divergence'] = (current_rej_eff - prev_rej_eff) * (current_efficiency - prev_efficiency)
    df['Bearish_Microstructure_Divergence'] = (prev_rej_eff - current_rej_eff) * (prev_efficiency - current_efficiency)
    
    # Microstructure Pattern Validation
    df['Rejection_Volume_Coherence'] = (df['Upper_Rejection'] - df['Lower_Rejection']) * df['volume'] / df['volume'].shift(1)
    
    efficiency_change = current_efficiency - prev_efficiency
    price_change_sign = np.sign(df['close'] - df['close'].shift(1))
    df['Efficiency_Momentum_Alignment'] = np.sign(efficiency_change) * price_change_sign
    
    # Composite Alpha Construction
    # Core Microstructure Signals
    df['Primary_Rejection_Momentum'] = df['Directional_Rejection_Pressure'] * df['Volume_Confirmed_Rejection']
    df['Regime_Transition_Signal'] = df['High_Volatility_Rejection'] * df['Volume_Surge_Rejection']
    df['Efficiency_Enhancement'] = df['Gap_Rejection_Efficiency'] * df['Opening_Efficiency_Momentum']
    
    # Validation Layers
    df['Microstructure_Coherence'] = df['Rejection_Volume_Coherence'] * df['Efficiency_Momentum_Alignment']
    df['Pattern_Confidence'] = df['Volatility_Rejection_Persistence'] * df['Volume_Rejection_Regime']
    
    # Final Alpha Factor
    alpha = (
        df['Primary_Rejection_Momentum'] * 0.3 +
        df['Regime_Transition_Signal'] * 0.25 +
        df['Efficiency_Enhancement'] * 0.2 +
        df['Microstructure_Coherence'] * 0.15 +
        df['Pattern_Confidence'] * 0.1
    )
    
    return alpha
