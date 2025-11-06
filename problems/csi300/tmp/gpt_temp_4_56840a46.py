import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum Factor
    Combines price momentum, volume momentum, and divergence signals to generate alpha factor
    """
    
    # Calculate Price Momentum Components
    df['price_momentum_5d'] = df['close'].pct_change(5)
    df['price_momentum_10d'] = df['close'].pct_change(10)
    df['price_acceleration'] = df['price_momentum_5d'] - df['price_momentum_10d']
    
    # Calculate Volume Momentum Components
    df['volume_roc_5d'] = df['volume'].pct_change(5)
    df['volume_roc_10d'] = df['volume'].pct_change(10)
    df['volume_acceleration'] = df['volume_roc_5d'] - df['volume_roc_10d']
    
    # Calculate Volume Persistence
    volume_direction = np.sign(df['volume_roc_5d'])
    volume_persistence = volume_direction.rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and not pd.isna(x.iloc[i])]), 
        raw=False
    )
    df['volume_persistence'] = volume_persistence
    
    # Calculate Price Persistence
    price_direction = np.sign(df['price_momentum_5d'])
    price_persistence = price_direction.rolling(window=5).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and not pd.isna(x.iloc[i])]), 
        raw=False
    )
    df['price_persistence'] = price_persistence
    
    # Compute Divergence Signals
    # Price-Volume Direction Divergence
    price_volume_divergence = np.zeros(len(df))
    for i in range(len(df)):
        if not (pd.isna(df['price_momentum_5d'].iloc[i]) or pd.isna(df['volume_roc_5d'].iloc[i])):
            if df['price_momentum_5d'].iloc[i] > 0 and df['volume_roc_5d'].iloc[i] < 0:
                # Bearish divergence
                price_volume_divergence[i] = -abs(df['price_momentum_5d'].iloc[i] * df['volume_roc_5d'].iloc[i])
            elif df['price_momentum_5d'].iloc[i] < 0 and df['volume_roc_5d'].iloc[i] > 0:
                # Bullish divergence
                price_volume_divergence[i] = abs(df['price_momentum_5d'].iloc[i] * df['volume_roc_5d'].iloc[i])
    
    df['price_volume_divergence'] = price_volume_divergence
    
    # Momentum Acceleration Divergence
    acceleration_divergence = np.zeros(len(df))
    for i in range(len(df)):
        if not (pd.isna(df['price_acceleration'].iloc[i]) or pd.isna(df['volume_acceleration'].iloc[i])):
            if df['price_acceleration'].iloc[i] > 0 and df['volume_acceleration'].iloc[i] < 0:
                # Weakening uptrend
                acceleration_divergence[i] = -abs(df['price_acceleration'].iloc[i] * df['volume_acceleration'].iloc[i])
            elif df['price_acceleration'].iloc[i] < 0 and df['volume_acceleration'].iloc[i] > 0:
                # Strengthening downtrend
                acceleration_divergence[i] = abs(df['price_acceleration'].iloc[i] * df['volume_acceleration'].iloc[i])
    
    df['acceleration_divergence'] = acceleration_divergence
    
    # Persistence Divergence
    persistence_divergence = np.zeros(len(df))
    for i in range(len(df)):
        if not (pd.isna(df['price_persistence'].iloc[i]) or pd.isna(df['volume_persistence'].iloc[i])):
            persistence_ratio = df['price_persistence'].iloc[i] / (df['volume_persistence'].iloc[i] + 1e-8)
            if persistence_ratio > 1.5:  # High price persistence, low volume persistence
                persistence_divergence[i] = -persistence_ratio
            elif persistence_ratio < 0.67:  # Low price persistence, high volume persistence
                persistence_divergence[i] = persistence_ratio
    
    df['persistence_divergence'] = persistence_divergence
    
    # Combine Divergence Components
    # Normalize components
    df['primary_divergence'] = df['price_volume_divergence'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 0, raw=False
    )
    
    df['acceleration_adjustment'] = df['acceleration_divergence'].rolling(window=20).apply(
        lambda x: 1 + (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 1, raw=False
    )
    
    df['persistence_multiplier'] = df['persistence_divergence'].rolling(window=20).apply(
        lambda x: 1 + (x.iloc[-1] - x.mean()) / (x.std() + 1e-8) if x.std() > 0 else 1, raw=False
    )
    
    # Final factor calculation
    factor = df['primary_divergence'] * df['acceleration_adjustment'] * df['persistence_multiplier']
    
    # Clean up intermediate columns
    cols_to_drop = ['price_momentum_5d', 'price_momentum_10d', 'price_acceleration', 
                   'volume_roc_5d', 'volume_roc_10d', 'volume_acceleration',
                   'volume_persistence', 'price_persistence', 'price_volume_divergence',
                   'acceleration_divergence', 'persistence_divergence', 'primary_divergence',
                   'acceleration_adjustment', 'persistence_multiplier']
    
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    return factor
