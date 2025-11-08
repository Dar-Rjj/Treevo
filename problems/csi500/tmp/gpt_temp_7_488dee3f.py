import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Volatility-Regime Adaptive Momentum with Volume-Price Alignment factor
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Calculation
    df['momentum_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['momentum_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['momentum_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Volatility-Regime Classification
    df['daily_return'] = df['close'].pct_change()
    df['volatility_20d'] = df['daily_return'].rolling(window=20, min_periods=10).std()
    df['volatility_median_60d'] = df['volatility_20d'].rolling(window=60, min_periods=30).median()
    
    # Regime Determination
    df['high_vol_regime'] = df['volatility_20d'] > df['volatility_median_60d']
    
    # Regime-Adaptive Momentum Weighting
    def calculate_weighted_momentum(row):
        if row['high_vol_regime']:
            # High volatility: emphasize short-term
            return (0.5 * row['momentum_3d'] + 
                    0.3 * row['momentum_10d'] + 
                    0.2 * row['momentum_20d'])
        else:
            # Low volatility: balanced approach
            return (0.3 * row['momentum_3d'] + 
                    0.4 * row['momentum_10d'] + 
                    0.3 * row['momentum_20d'])
    
    df['weighted_momentum'] = df.apply(calculate_weighted_momentum, axis=1)
    
    # Volatility adjustment
    df['weighted_momentum_vol_adj'] = df['weighted_momentum'] / df['volatility_20d']
    
    # Volume-Price Alignment Analysis
    df['volume_ma_5d'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['volume_ma_20d'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_ratio'] = df['volume_ma_5d'] / df['volume_ma_20d']
    
    def get_volume_confirmation(row):
        if pd.isna(row['weighted_momentum_vol_adj']) or pd.isna(row['volume_ratio']):
            return 1.0
        
        momentum_positive = row['weighted_momentum_vol_adj'] > 0
        
        if momentum_positive:
            if row['volume_ratio'] > 1.1:
                return 1.3  # Strong bullish confirmation
            else:
                return 0.7  # Weak bullish confirmation
        else:
            if row['volume_ratio'] < 0.9:
                return 1.3  # Strong bearish confirmation
            else:
                return 0.7  # Weak bearish confirmation
    
    df['volume_confirmation'] = df.apply(get_volume_confirmation, axis=1)
    
    # Intraday Strength Component
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['range_efficiency'] = df['range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate persistence streak
    df['efficiency_sign'] = np.sign(df['range_efficiency'])
    df['streak'] = 0
    current_streak = 0
    current_sign = 0
    
    for i in range(len(df)):
        if pd.isna(df['efficiency_sign'].iloc[i]):
            df.loc[df.index[i], 'streak'] = 0
            current_streak = 0
            current_sign = 0
        elif df['efficiency_sign'].iloc[i] == current_sign and current_sign != 0:
            current_streak += 1
            df.loc[df.index[i], 'streak'] = current_streak
        else:
            current_sign = df['efficiency_sign'].iloc[i]
            current_streak = 1 if current_sign != 0 else 0
            df.loc[df.index[i], 'streak'] = current_streak
    
    df['persistence_bonus'] = np.minimum(df['streak'] * 0.1, 0.5)
    df['intraday_strength'] = df['range_efficiency'] + df['persistence_bonus']
    
    # Final Alpha Factor Construction
    df['alpha_factor'] = (df['weighted_momentum_vol_adj'] * df['volume_confirmation'] + 
                          df['intraday_strength'])
    
    # Clean up intermediate columns
    result = df['alpha_factor'].copy()
    
    return result
