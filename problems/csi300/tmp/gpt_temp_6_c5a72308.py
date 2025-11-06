import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Dynamic Regime Detection
    # Volatility Regime Classification
    df['short_term_vol'] = df['true_range'].rolling(window=3, min_periods=3).mean()
    df['medium_term_vol'] = df['true_range'].rolling(window=10, min_periods=10).mean()
    df['vol_regime'] = np.where(df['short_term_vol'] / df['medium_term_vol'] > 1.2, 'High', 'Low')
    
    # Trend Regime Classification
    df['price_trend'] = (df['close'] - df['close'].shift(5)) / 5
    df['volume_trend'] = (df['volume'] - df['volume'].shift(5)) / 5
    df['trend_regime'] = np.where(abs(df['price_trend']) > 0.02, 'Strong', 'Weak')
    
    # Adaptive Momentum Framework
    # Multi-timeframe Momentum Components
    df['momentum_ultra'] = df['close'] / df['close'].shift(1) - 1
    df['momentum_short'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_medium'] = df['close'] / df['close'].shift(6) - 1
    df['momentum_long'] = df['close'] / df['close'].shift(12) - 1
    
    # Regime-Adaptive Smoothing
    def adaptive_momentum_smoothing(row):
        if row['vol_regime'] == 'High':
            # 2-day exponential weighting: 0.7 × Current + 0.3 × Previous
            weights = [0.7, 0.3]
            components = [row['momentum_ultra'], row['momentum_short']]
        else:
            # 5-day exponential weighting: [0.4, 0.3, 0.15, 0.1, 0.05]
            weights = [0.4, 0.3, 0.15, 0.1, 0.05]
            components = [row['momentum_ultra'], row['momentum_short'], 
                         row['momentum_medium'], row['momentum_long'], row['momentum_long']]
        
        return np.dot(weights[:len(components)], components[:len(weights)])
    
    df['smoothed_momentum'] = df.apply(adaptive_momentum_smoothing, axis=1)
    
    # Volume Acceleration with Dynamic Periods
    df['volume_change'] = df['volume'] / df['volume'].shift(1)
    df['volume_acceleration'] = df['volume_change'] / df['volume_change'].shift(1)
    
    # Adaptive Smoothing for Volume Acceleration
    def adaptive_volume_smoothing(row, df_window):
        if row['trend_regime'] == 'Strong':
            # 2-day simple average
            return df_window['volume_acceleration'].iloc[-2:].mean()
        else:
            # 4-day simple average
            return df_window['volume_acceleration'].iloc[-4:].mean()
    
    df['smoothed_acceleration'] = np.nan
    for i in range(4, len(df)):
        window = df.iloc[max(0, i-10):i+1]
        current_row = df.iloc[i]
        df.loc[df.index[i], 'smoothed_acceleration'] = adaptive_volume_smoothing(current_row, window)
    
    # Momentum-Volume Trend Alignment
    df['momentum_direction'] = np.sign(df['smoothed_momentum'])
    df['volume_trend_direction'] = np.sign(df['volume_trend'])
    df['alignment'] = df['momentum_direction'] * df['volume_trend_direction']
    
    df['relative_strength'] = abs(df['smoothed_momentum']) * abs(df['volume_trend'])
    df['confirmation_score'] = df['alignment'] * df['relative_strength']
    
    # Dynamic Volatility Scaling
    df['price_volatility'] = abs(df['close'] - df['close'].shift(1))
    df['range_volatility'] = df['high'] - df['low']
    df['gap_volatility'] = abs(df['open'] - df['prev_close'])
    
    df['base_scaling'] = 1 / (df['price_volatility'] + df['range_volatility'] + df['gap_volatility'])
    
    df['vol_multiplier'] = np.where(df['vol_regime'] == 'High', 0.8, 1.2)
    df['scaled_volatility'] = df['base_scaling'] * df['vol_multiplier']
    
    # Final Alpha Construction
    df['core_signal'] = df['smoothed_momentum'] * df['smoothed_acceleration']
    df['alignment_enhanced'] = df['core_signal'] * df['confirmation_score']
    df['final_alpha'] = df['alignment_enhanced'] * df['scaled_volatility']
    
    # Clean up intermediate columns
    result = df['final_alpha'].copy()
    
    return result
