import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Factors
    df['1_day_return'] = df['close'] / df['close'].shift(1) - 1
    df['3_day_return'] = df['close'] / df['close'].shift(3) - 1
    df['5_day_return'] = df['close'] / df['close'].shift(5) - 1
    
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['gap_behavior'] = df['open'] / df['close'].shift(1) - 1
    
    df['return_consistency'] = df['1_day_return'].rolling(window=3).apply(lambda x: (x > 0).sum())
    df['acceleration'] = df['3_day_return'] - df['5_day_return']
    df['direction_persistence'] = (np.sign(df['1_day_return']) == np.sign(df['3_day_return'])).astype(float)
    
    # Volume Dynamics
    df['volume_momentum'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(1) - 1) - (df['volume'].shift(1) / df['volume'].shift(2) - 1)
    df['volume_stability'] = df['volume'] / df['volume'].shift(5) - 1
    
    df['volume_confirmed_return'] = df['1_day_return'] * df['volume_momentum']
    df['volume_range_synergy'] = df['daily_range'] * df['volume_momentum']
    df['gap_volume_alignment'] = df['gap_behavior'] * df['volume_momentum']
    
    df['volume_breakout'] = (df['volume'] > 1.5 * df['volume'].shift(3)).astype(float)
    df['volume_contraction'] = (df['volume'] < 0.8 * df['volume'].shift(3)).astype(float)
    df['volume_stability_flag'] = (abs(df['volume_stability']) < 0.2).astype(float)
    
    # Volatility Framework
    df['3_day_volatility'] = df['1_day_return'].rolling(window=3).std()
    df['5_day_volatility'] = df['1_day_return'].rolling(window=5).std()
    df['volatility_ratio'] = df['3_day_volatility'] / (df['5_day_volatility'] + 1e-8)
    
    df['volatility_scaled_1_day_return'] = df['1_day_return'] / (df['3_day_volatility'] + 0.001)
    df['volatility_scaled_3_day_return'] = df['3_day_return'] / (df['3_day_volatility'] + 0.001)
    df['volatility_scaled_5_day_return'] = df['5_day_return'] / (df['5_day_volatility'] + 0.001)
    
    df['rising_volatility'] = (df['volatility_ratio'] > 1.1).astype(float)
    df['falling_volatility'] = (df['volatility_ratio'] < 0.9).astype(float)
    df['stable_volatility'] = ((df['volatility_ratio'] >= 0.95) & (df['volatility_ratio'] <= 1.05)).astype(float)
    
    # Simple Composite Factors
    df['clean_momentum'] = df['3_day_return'] / (df['3_day_volatility'] + 0.001)
    df['persistent_clean_momentum'] = df['clean_momentum'] * df['return_consistency']
    df['direction_persistent_momentum'] = df['clean_momentum'] * df['direction_persistence']
    
    df['volume_confirmed_momentum'] = df['3_day_return'] * df['volume_momentum']
    df['volume_confirmed_volatility_return'] = df['volatility_scaled_3_day_return'] * df['volume_momentum']
    df['volume_stable_momentum'] = df['clean_momentum'] * df['volume_stability_flag']
    
    df['falling_volatility_momentum'] = df['falling_volatility'] * df['volatility_scaled_3_day_return']
    df['stable_volatility_quality'] = df['stable_volatility'] * df['clean_momentum']
    df['rising_volatility_volume'] = df['rising_volatility'] * df['volume_confirmed_momentum']
    
    # Advanced Composites
    df['volume_volatility_composite'] = df['volume_confirmed_volatility_return'] / (df['3_day_volatility'] + 0.001)
    df['volume_stable_volatility_momentum'] = df['volume_stable_momentum'] * df['stable_volatility']
    df['falling_volatility_volume_premium'] = df['falling_volatility_momentum'] * df['volume_momentum']
    
    df['persistent_volume_momentum'] = df['volume_confirmed_momentum'] * df['return_consistency']
    df['direction_persistent_volatility_return'] = df['volatility_scaled_3_day_return'] * df['direction_persistence']
    df['multi_persistence_momentum'] = df['persistent_clean_momentum'] * df['direction_persistence']
    
    df['volume_volatility_regime'] = df['volume_volatility_composite'] * df['stable_volatility']
    df['falling_volatility_persistence'] = df['falling_volatility_momentum'] * df['return_consistency']
    df['stable_regime_quality'] = df['volume_stable_volatility_momentum'] * df['direction_persistence']
    
    # Final Alpha Output - Composite of selected factors
    alpha = (df['persistent_clean_momentum'] + 
             df['volume_volatility_composite'] + 
             df['falling_volatility_persistence'] + 
             df['stable_regime_quality']) / 4
    
    return alpha
