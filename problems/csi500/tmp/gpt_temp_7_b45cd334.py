import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Components
    df['intraday_momentum'] = df['close'] - df['open']
    df['daily_range'] = df['high'] - df['low']
    df['return_1d'] = df['close'].diff(1)
    df['return_3d'] = df['close'] - df['close'].shift(3)
    
    # Volume Components
    df['volume_change'] = df['volume'].diff(1)
    df['volume_momentum_sign'] = np.sign(df['intraday_momentum']) * np.sign(df['volume_change'])
    
    # Calculate Positive Alignment Days (consecutive days with positive volume_momentum_sign)
    df['positive_alignment_flag'] = (df['volume_momentum_sign'] > 0).astype(int)
    df['positive_alignment_days'] = 0
    
    current_streak = 0
    for i in range(len(df)):
        if df['positive_alignment_flag'].iloc[i] == 1:
            current_streak += 1
        else:
            current_streak = 0
        df['positive_alignment_days'].iloc[i] = current_streak
    
    # Absolute Momentum Construction
    df['short_term_momentum'] = np.abs(df['intraday_momentum'])
    df['medium_term_momentum'] = df['short_term_momentum'].rolling(window=3, min_periods=1).sum()
    df['combined_momentum'] = (2 * df['short_term_momentum'] + df['medium_term_momentum']) / 3
    
    # Volatility Context
    df['recent_volatility'] = df['daily_range'].rolling(window=3, min_periods=1).mean()
    df['baseline_volatility'] = df['daily_range'].rolling(window=10, min_periods=1).mean()
    df['volatility_ratio'] = df['recent_volatility'] / df['baseline_volatility']
    
    # Factor Integration
    df['base_signal'] = df['combined_momentum'] * np.sign(df['intraday_momentum'])
    
    # Volume Boost: 1 + 0.1 × Positive_Alignment_Days (when Volume_Momentum_Sign > 0)
    df['volume_boost'] = 1 + 0.1 * df['positive_alignment_days'] * (df['volume_momentum_sign'] > 0).astype(float)
    
    # Volatility Adjustment
    df['volatility_adjustment'] = df['base_signal'] / df['volatility_ratio']
    
    # Final Alpha
    df['raw_alpha'] = df['volatility_adjustment'] * df['volume_boost']
    df['final_alpha'] = np.sign(df['raw_alpha'])
    
    return df['final_alpha']
