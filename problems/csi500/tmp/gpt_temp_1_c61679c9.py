import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Momentum with Volume Convergence alpha factor
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Alpha factor values indexed by date
    """
    
    # Price Momentum Components
    df['daily_change'] = df['close'] - df['close'].shift(1)
    df['intraday_momentum'] = df['close'] - df['open']
    df['price_range'] = df['high'] - df['low']
    
    # Multi-Timeframe Construction
    # Short-Term (3 days)
    df['momentum_3d'] = df['close'] - df['close'].shift(2)
    df['range_3d'] = (df['high'] - df['low']) + (df['high'].shift(1) - df['low'].shift(1)) + (df['high'].shift(2) - df['low'].shift(2))
    df['volume_3d'] = df['volume'] + df['volume'].shift(1) + df['volume'].shift(2)
    
    # Medium-Term (10 days)
    df['momentum_10d'] = df['close'] - df['close'].shift(9)
    
    # Calculate rolling sums for range and volume
    df['range_10d'] = (df['high'] - df['low']).rolling(window=10, min_periods=1).sum()
    df['volume_10d'] = df['volume'].rolling(window=10, min_periods=1).sum()
    
    # Volatility Scaling
    df['vol_short'] = df['range_3d'] / 3
    df['vol_medium'] = df['range_10d'] / 10
    
    # Volatility-Scaled Momentum
    df['vsm_3d'] = df['momentum_3d'] / df['range_3d'].replace(0, np.nan)
    df['vsm_10d'] = df['momentum_10d'] / df['range_10d'].replace(0, np.nan)
    
    # Volatility Regime
    df['vol_ratio'] = df['vol_short'] / df['vol_medium'].replace(0, np.nan)
    
    # Volume-Price Convergence
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_direction'] = np.sign(df['volume_change'])
    
    # Calculate volume streak (consecutive same direction days)
    df['volume_streak'] = 0
    for i in range(1, len(df)):
        if df['volume_direction'].iloc[i] == df['volume_direction'].iloc[i-1]:
            df['volume_streak'].iloc[i] = df['volume_streak'].iloc[i-1] + 1
        else:
            df['volume_streak'].iloc[i] = 1
    
    # Momentum-Volume Alignment
    df['direction_match'] = np.sign(df['daily_change']) * df['volume_direction']
    
    # Calculate alignment streak (consecutive positive match days)
    df['alignment_streak'] = 0
    for i in range(1, len(df)):
        if df['direction_match'].iloc[i] > 0 and df['direction_match'].iloc[i-1] > 0:
            df['alignment_streak'].iloc[i] = df['alignment_streak'].iloc[i-1] + 1
        else:
            df['alignment_streak'].iloc[i] = 1 if df['direction_match'].iloc[i] > 0 else 0
    
    df['convergence_strength'] = df['alignment_streak'] * abs(df['daily_change'])
    
    # Volume Regime
    df['volume_ratio'] = df['volume_3d'] / df['volume_10d'].replace(0, np.nan)
    
    # Factor Integration
    # Base Momentum Signal
    df['base_momentum'] = (2 * df['vsm_3d'] + df['vsm_10d']) / 3
    df['volume_weighted'] = df['base_momentum'] * np.log(1 + df['volume'])
    
    # Convergence Enhancement
    df['volume_confirmed'] = df['volume_weighted'] * (1 + 0.1 * df['alignment_streak'])
    df['trend_enhanced'] = df['volume_confirmed'] * (1 + 0.05 * df['volume_streak'])
    
    # Volatility Adjustment
    df['volatility_adjusted'] = df['trend_enhanced'].copy()
    high_vol_mask = df['vol_ratio'] > 1.1
    low_vol_mask = df['vol_ratio'] < 0.9
    df.loc[high_vol_mask, 'volatility_adjusted'] = df.loc[high_vol_mask, 'trend_enhanced'] * 0.7
    df.loc[low_vol_mask, 'volatility_adjusted'] = df.loc[low_vol_mask, 'trend_enhanced'] * 1.3
    
    # Volume Regime Scaling
    df['final_factor'] = df['volatility_adjusted'].copy()
    high_volume_mask = df['volume_ratio'] > 1.05
    low_volume_mask = df['volume_ratio'] < 0.95
    df.loc[high_volume_mask, 'final_factor'] = df.loc[high_volume_mask, 'volatility_adjusted'] * 1.2
    df.loc[low_volume_mask, 'final_factor'] = df.loc[low_volume_mask, 'volatility_adjusted'] * 0.8
    
    return df['final_factor']
