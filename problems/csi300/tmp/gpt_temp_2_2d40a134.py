import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Entropic-Microstructure Regime Adaptive Alpha Dynamics
    """
    # Calculate True Range
    df['TR'] = np.maximum(df['high'] - df['low'], 
                         np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    
    # Calculate rolling True Range
    df['TR2'] = df['TR'].rolling(window=2).mean()
    df['TR10'] = df['TR'].rolling(window=10).mean()
    
    # Calculate Net Tick Pressure (simplified as price-volume pressure)
    df['Net_Tick_Pressure'] = (df['close'] - df['open']) * df['volume']
    
    # Entropic Volatility State components
    df['Entropic_Vol_Expansion'] = ((df['high'] - df['low']) / 
                                   (df['high'].shift(5) - df['low'].shift(5) + 1e-8)) * (df['TR10'] / (df['TR2'] + 1e-8))
    
    df['Multi_Scale_Entropic_Accel'] = ((df['high'] - df['low']) / (df['high'].shift(3) - df['low'].shift(3) + 1e-8)) - \
                                      ((df['high'].shift(8) - df['low'].shift(8)) / (df['high'].shift(15) - df['low'].shift(15) + 1e-8))
    
    # Entropic Volatility Persistence (correlation over 8 days)
    df['price_change'] = df['close'] - df['open']
    corr_window = 8
    entropic_persistence = []
    for i in range(len(df)):
        if i >= corr_window:
            window_data = df.iloc[i-corr_window+1:i+1]
            corr_val = window_data['price_change'].corr(window_data['Net_Tick_Pressure'])
            entropic_persistence.append(corr_val * (df['TR10'].iloc[i] / (df['TR10'].shift(3).iloc[i] + 1e-8)))
        else:
            entropic_persistence.append(0)
    df['Entropic_Vol_Persistence'] = entropic_persistence
    
    # Microstructure-Entropic State components
    df['Microstructure_Entropic_Efficiency'] = ((df['close'] - df['close'].shift(5)) / 
                                               (df['amount'] + df['amount'].shift(1) + df['amount'].shift(2) + 1e-8)) * df['Net_Tick_Pressure']
    
    df['Microstructure_Entropic_Entanglement'] = ((df['close'] - df['close'].shift(2)) * (1 / (df['amount'] + 1e-8)) * 
                                                 df['Net_Tick_Pressure'] / (df['volume'] + 1e-8))
    
    df['Microstructure_Entropic_Persistence'] = (np.sign(df['close'] - df['close'].shift(2)) * 
                                                np.sign(df['close'].shift(2) - df['close'].shift(4)) * 
                                                (df['close'] - df['close'].shift(4)) * (df['TR10'] / (df['TR2'] + 1e-8)))
    
    # Entropic Pressure-Microstructure Interaction
    df['Entropic_Pressure_Efficiency'] = ((df['close'] - df['close'].shift(5)) / 
                                         (df['Net_Tick_Pressure'] + df['Net_Tick_Pressure'].shift(2) + df['Net_Tick_Pressure'].shift(4) + 1e-8))
    
    df['Microstructure_Pressure_Decoherence'] = (abs(df['close'] - df['close'].shift(2)) / 
                                                ((df['Net_Tick_Pressure'] / (df['Net_Tick_Pressure'].shift(2) + 1e-8)) + 1e-8) * 
                                                (df['amount'] / (df['amount'].shift(2) + 1e-8)))
    
    df['Entropic_Microstructure_Compression'] = (df['Net_Tick_Pressure'] / (df['TR10'] + 1e-8)) * (1 / (df['amount'] + 1e-8))
    
    # Entropic Breakout Momentum System
    df['Entropic_Gap_Fill_Efficiency'] = np.where(df['open'] > df['close'].shift(2),
                                                 (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8),
                                                 (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)) * df['Net_Tick_Pressure']
    
    df['Entropic_Volume_Price_Divergence'] = ((df['close'] - df['close'].shift(2)) / (df['high'].shift(2) - df['low'].shift(2) + 1e-8) - 
                                             (df['amount'] - df['amount'].shift(2)) / (df['amount'].shift(2) + 1e-8)) * (df['TR10'] / (df['TR2'] + 1e-8))
    
    df['Entropic_Intraday_Strength'] = ((df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * 
                                       (df['close'].shift(2) - df['open'].shift(2)) / (df['high'].shift(2) - df['low'].shift(2) + 1e-8) * 
                                       df['Net_Tick_Pressure'])
    
    # Multi-Scale Entropic Momentum
    df['Entropic_Momentum_Persistence'] = ((df['close'] - df['close'].shift(2)) * 
                                          (df['Net_Tick_Pressure'] - df['Net_Tick_Pressure'].shift(2)) * 
                                          (df['close'] / (df['close'].shift(5) + 1e-8) - 1))
    
    df['Microstructure_Entropic_Wave'] = ((df['close'] - df['low']) / (df['TR10'] + 1e-8) * 
                                         df['Net_Tick_Pressure'] * (1 / (df['amount'] + 1e-8)))
    
    df['Entropic_Flow_Interference'] = (((df['volume'] - df['volume'].shift(2)) / 
                                        (df['Net_Tick_Pressure'] - df['Net_Tick_Pressure'].shift(2) + 1e-8)) * 
                                       (df['amount'] / df['amount'].rolling(window=6).mean()) * 
                                       (df['TR10'] / (df['TR2'] + 1e-8)))
    
    # Entropic Breakout Validation
    df['Entropic_Volatility_Momentum_Alignment'] = (((df['close'] - df['close'].shift(5)) / (df['high'].shift(5) - df['low'].shift(5) + 1e-8)) * 
                                                   ((df['high'] - df['low']) / (df['high'].shift(5) - df['low'].shift(5) + 1e-8)) * 
                                                   (df['TR10'] / (df['TR2'] + 1e-8)))
    
    df['Entropic_Microstructure_Regime_Stability'] = (df['amount'] / ((df['amount'].shift(2) + df['amount'].shift(4) + df['amount'].shift(6)) / 3 + 1e-8)) * df['Net_Tick_Pressure']
    
    df['Entropic_Pressure_Coherence'] = (((df['Net_Tick_Pressure'] - df['Net_Tick_Pressure'].shift(5)) / 
                                         (df['Net_Tick_Pressure'].shift(5) - df['Net_Tick_Pressure'].shift(10) + 1e-8)) * 
                                        (df['amount'] / (df['amount'].shift(5) + 1e-8)))
    
    # Adaptive Entropic-Microstructure Alpha Construction
    df['Regime_Adaptive_Entropic_Momentum'] = (df['Entropic_Vol_Expansion'] * 
                                              df['Microstructure_Entropic_Efficiency'] * 
                                              df['Entropic_Momentum_Persistence'])
    
    df['Entropic_Breakout_Confirmation'] = (df['Entropic_Gap_Fill_Efficiency'] * 
                                           df['Entropic_Volume_Price_Divergence'] * 
                                           df['Entropic_Intraday_Strength'])
    
    df['Microstructure_Entropic_Microstructure_Alpha'] = (df['Entropic_Pressure_Efficiency'] * 
                                                         df['Microstructure_Pressure_Decoherence'] * 
                                                         df['Entropic_Flow_Interference'])
    
    # Dynamic Entropic-Microstructure Integration
    # Calculate regime indicators
    high_entropic_vol_regime = df['Entropic_Vol_Expansion'].rolling(window=5).mean() > df['Entropic_Vol_Expansion'].rolling(window=20).mean()
    low_microstructure_state = df['Microstructure_Entropic_Efficiency'].rolling(window=5).mean() < df['Microstructure_Entropic_Efficiency'].rolling(window=20).mean()
    
    # Final alpha factor with regime adaptation
    alpha = (high_entropic_vol_regime * df['Regime_Adaptive_Entropic_Momentum'] +
             low_microstructure_state * df['Microstructure_Entropic_Microstructure_Alpha'] +
             (~high_entropic_vol_regime & ~low_microstructure_state) * df['Entropic_Breakout_Confirmation'])
    
    # Apply entropic microstructure compression filtering
    compression_threshold = df['Entropic_Microstructure_Compression'].rolling(window=10).mean()
    alpha = alpha * (df['Entropic_Microstructure_Compression'] > compression_threshold)
    
    # Normalize and clean
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = (alpha - alpha.rolling(window=20).mean()) / (alpha.rolling(window=20).std() + 1e-8)
    
    return alpha
