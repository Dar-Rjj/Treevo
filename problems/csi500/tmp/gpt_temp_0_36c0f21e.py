import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Momentum-Volume Divergence Analysis
    df['momentum_divergence'] = (df['close'] / df['close'].shift(1) - 1) - (df['volume'] / df['volume'].shift(1) - 1)
    df['acceleration_divergence'] = df['momentum_divergence'] - df['momentum_divergence'].shift(1)
    df['volume_intensity'] = df['volume'] / df['amount'] * (df['high'] - df['low'])
    
    # Fractal Market Microstructure
    df['fractal_range_ratio'] = (df['high'] - df['low']) / ((df['high'].shift(1) - df['low'].shift(1) + df['high'].shift(2) - df['low'].shift(2)) / 2)
    df['microstructure_efficiency'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])) * np.log(df['volume'] / df['volume'].shift(1))
    
    # Fractal Persistence
    df['fractal_persistence'] = 0
    count = 0
    for i in range(len(df)):
        if i >= 2 and df['fractal_range_ratio'].iloc[i] > 1.1:
            count += 1
        else:
            count = 0
        df.iloc[i, df.columns.get_loc('fractal_persistence')] = count
    
    # Market Regime Classification
    df['market_regime'] = 1.0
    for i in range(len(df)):
        if i >= 1:
            if df['fractal_range_ratio'].iloc[i] > 1.2 and df['volume_intensity'].iloc[i] > df['volume_intensity'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('market_regime')] = 1.3
            elif df['fractal_range_ratio'].iloc[i] < 0.9 and abs(df['close'].iloc[i] - df['open'].iloc[i]) < 0.3 * (df['high'].iloc[i] - df['low'].iloc[i]):
                df.iloc[i, df.columns.get_loc('market_regime')] = 0.7
    
    # Regime Persistence Multiplier
    df['regime_persistence_multiplier'] = 1 + (df['fractal_persistence'] / 15)
    
    # Multi-Scale Divergence Patterns
    df['intraday_divergence'] = ((df['close'] - df['open']) / (df['high'] - df['low'])) - ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1))
    df['short_term_divergence'] = (df['close'] / df['close'].shift(5) - 1) - (df['volume'] / df['volume'].shift(5) - 1)
    df['divergence_acceleration'] = df['short_term_divergence'] - df['intraday_divergence']
    
    # Volume Microstructure Features
    df['volume_clustering'] = 0
    for i in range(len(df)):
        if i >= 5:
            count_cluster = 0
            for j in range(5):
                if (df['volume'].iloc[i-j] > df['volume'].iloc[i-j-1]) and (df['close'].iloc[i-j] > df['close'].iloc[i-j-1]):
                    count_cluster += 1
            df.iloc[i, df.columns.get_loc('volume_clustering')] = count_cluster
    
    df['microstructure_strength'] = (df['volume'] / df['volume'].shift(1)) * ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)))
    
    # Final Alpha Construction
    df['core_divergence_signal'] = df['momentum_divergence'] * df['acceleration_divergence']
    df['fractal_enhancement'] = df['fractal_range_ratio'] * df['microstructure_efficiency']
    df['regime_modulation'] = df['market_regime'] * df['regime_persistence_multiplier']
    
    # Primary Factor
    df['alpha_factor'] = (df['core_divergence_signal'] * 
                         df['fractal_enhancement'] * 
                         df['divergence_acceleration'] * 
                         df['regime_modulation'] * 
                         (1 + df['volume_clustering'] / 8))
    
    return df['alpha_factor']
