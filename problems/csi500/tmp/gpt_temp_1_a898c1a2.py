import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Momentum
    df = df.copy()
    
    # Intraday Efficiency
    df['intraday_return'] = df['close'] - df['open']
    df['intraday_range'] = df['high'] - df['low']
    df['intraday_efficiency'] = df['intraday_return'] / df['intraday_range'].replace(0, np.nan)
    
    # Overnight Efficiency
    df['overnight_return'] = df['open'] - df['close'].shift(1)
    df['prev_day_range'] = (df['high'] - df['low']).shift(1)
    df['overnight_efficiency'] = df['overnight_return'] / df['prev_day_range'].replace(0, np.nan)
    
    # Combined Efficiency Signal
    df['combined_efficiency'] = (df['intraday_efficiency'] + df['overnight_efficiency']) / 2
    df['efficiency_direction'] = np.sign(df['combined_efficiency'])
    
    # Momentum Persistence Tracking
    df['persistence_streak'] = 1
    for i in range(1, len(df)):
        if df['efficiency_direction'].iloc[i] == df['efficiency_direction'].iloc[i-1]:
            df['persistence_streak'].iloc[i] = df['persistence_streak'].iloc[i-1] + 1
        else:
            df['persistence_streak'].iloc[i] = 1
    
    # Volume Confirmation Mechanism
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['volume_direction'] = np.sign(df['volume_ratio'] - 1)
    df['volume_streak'] = 1
    
    for i in range(1, len(df)):
        if df['volume_direction'].iloc[i] == df['volume_direction'].iloc[i-1]:
            df['volume_streak'].iloc[i] = df['volume_streak'].iloc[i-1] + 1
        else:
            df['volume_streak'].iloc[i] = 1
    
    # Volume-Price Alignment
    df['alignment'] = df['efficiency_direction'] == df['volume_direction']
    df['confirmation_strength'] = np.abs(df['combined_efficiency']) * df['volume_streak']
    df['confirmation_strength'] = np.where(df['alignment'], df['confirmation_strength'], -df['confirmation_strength'])
    
    # Adaptive Volatility Scaling
    df['short_term_vol'] = df['close'].rolling(window=5).std()
    df['medium_term_vol'] = df['close'].rolling(window=10).std()
    df['long_term_vol'] = df['close'].rolling(window=20).std()
    df['vol_regime_ratio'] = df['short_term_vol'] / df['long_term_vol']
    
    # Base Persistence Factor
    df['base_factor'] = df['persistence_streak'] * df['confirmation_strength'] * df['efficiency_direction']
    
    # Volatility Regime Adjustment
    conditions = [
        df['vol_regime_ratio'] > 1.2,  # High volatility - dampen
        df['vol_regime_ratio'] < 0.8,  # Low volatility - amplify
        (df['vol_regime_ratio'] >= 0.8) & (df['vol_regime_ratio'] <= 1.2)  # Neutral
    ]
    choices = [
        df['base_factor'] / df['vol_regime_ratio'],  # Dampen in high volatility
        df['base_factor'] * (1 / df['vol_regime_ratio']),  # Amplify in low volatility
        df['base_factor']  # No adjustment in neutral
    ]
    
    df['alpha_factor'] = np.select(conditions, choices, default=df['base_factor'])
    
    return df['alpha_factor']
