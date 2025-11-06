import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Structure
    df = df.copy()
    
    # Multi-Timeframe Returns
    df['ultra_short'] = df['close'] / df['close'].shift(1) - 1
    df['short_term'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term'] = df['close'] / df['close'].shift(8) - 1
    
    # Momentum Acceleration
    df['primary_acceleration'] = df['short_term'] - df['medium_term']
    
    # Momentum Quality - Direction consistency
    df['return_sign'] = np.sign(df['ultra_short'])
    df['direction_consistency'] = df['return_sign'].rolling(window=3).apply(
        lambda x: (x == x.iloc[0]).sum() if not x.isna().any() else np.nan
    )
    
    # Momentum Quality - Strength ratio
    df['strength_ratio'] = df['short_term'] / df['medium_term']
    
    # Volatility Framework - Range-Based Volatility
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['short_term_vol'] = df['daily_range'].rolling(window=3).std()
    df['medium_term_vol'] = df['daily_range'].rolling(window=8).std()
    df['volatility_ratio'] = df['short_term_vol'] / df['medium_term_vol']
    
    # Volatility-Adjusted Momentum
    df['risk_adjusted_return'] = df['ultra_short'] / df['short_term_vol'].replace(0, np.nan)
    
    # Volume Dynamics
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    df['volume_trend'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_acceleration'] = df['volume_change'] - df['volume_change'].shift(1)
    
    # Price-Volume Alignment
    df['confirmation'] = (np.sign(df['ultra_short']) == np.sign(df['volume_change'])).astype(float)
    df['divergence'] = (np.sign(df['ultra_short']) != np.sign(df['volume_change'])).astype(float)
    df['alignment_strength'] = np.abs(df['ultra_short']) * np.abs(df['volume_change'])
    
    # Volume persistence
    df['volume_persistence'] = (df['volume_change'] > 0).rolling(window=3).sum()
    
    # Simple Cross-Regime Interactions
    df['low_vol_momentum'] = df['risk_adjusted_return'] * (df['volatility_ratio'] < 0.8)
    df['high_vol_acceleration'] = df['primary_acceleration'] * (df['volatility_ratio'] > 1.2)
    df['stable_vol_trending'] = df['direction_consistency'] * (df['short_term_vol'] < 0.01)
    
    df['confirmed_trend'] = df['short_term'] * df['confirmation']
    df['divergence_signal'] = df['primary_acceleration'] * df['divergence']
    df['volume_persistence_momentum'] = df['strength_ratio'] * df['volume_persistence']
    
    df['full_alignment'] = ((df['direction_consistency'] == 3) & 
                           (df['confirmation'] == 1) & 
                           (df['volatility_ratio'] < 1)).astype(float)
    df['momentum_volume'] = ((df['primary_acceleration'] > 0) & 
                            (df['volume_change'] > 0)).astype(float)
    df['volatility_breakout'] = ((df['volatility_ratio'] > 1.5) & 
                                (df['ultra_short'] > 0)).astype(float)
    
    # Composite Alpha Construction
    # Core Components
    df['base_momentum'] = df['risk_adjusted_return']
    df['volume_adjusted'] = df['base_momentum'] * (1 + df['alignment_strength'])
    df['volatility_scaled'] = df['volume_adjusted'] * (2 - df['volatility_ratio'])
    
    # Interaction Overlays
    df['composite'] = df['volatility_scaled']
    df['regime_alignment'] = df['composite'] * df['full_alignment']
    df['acceleration_premium'] = df['composite'] + df['primary_acceleration']
    df['volume_persistence_adj'] = df['composite'] * (1 + df['volume_persistence'] / 3)
    
    # Final Alpha with regime interactions and volatility adjustment
    alpha = (df['acceleration_premium'] + 
             df['regime_alignment'] + 
             df['volume_persistence_adj'] + 
             df['low_vol_momentum'] + 
             df['high_vol_acceleration'] + 
             df['confirmed_trend'])
    
    return alpha
