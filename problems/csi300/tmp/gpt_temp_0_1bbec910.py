import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Trend Persistence Alpha with Volume-Range Confirmation
    Generates alpha factor based on multi-timeframe trend analysis, volume confirmation,
    trend persistence measurement, and intraday range consistency.
    """
    df = data.copy()
    
    # Multi-Timeframe Trend Analysis
    df['trend_3d'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['trend_8d'] = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    df['trend_consistency'] = np.sign(df['trend_3d']) * np.sign(df['trend_8d'])
    
    # Volume-Price Alignment Framework
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_trend'] = df['volume'] / df['volume_ma_5']
    
    # Volume alignment confidence scoring
    conditions = [
        (df['volume_trend'] > 1.1) & (np.sign(df['volume_trend'] - 1) == np.sign(df['trend_8d'])),
        (df['volume_trend'].between(0.9, 1.1)) & (np.sign(df['volume_trend'] - 1) == np.sign(df['trend_8d'])),
        True  # default case (low confidence)
    ]
    choices = [1.2, 1.0, 0.6]
    df['volume_confidence'] = np.select(conditions, choices, default=0.6)
    
    # Trend Persistence Measurement
    # Direction persistence score
    df['trend_direction'] = np.sign(df['trend_8d'])
    persistence_count = []
    current_count = 0
    current_direction = 0
    
    for i, direction in enumerate(df['trend_direction']):
        if direction == current_direction and not pd.isna(direction):
            current_count += 1
        else:
            current_count = 1 if not pd.isna(direction) else 0
            current_direction = direction
        persistence_count.append(current_count)
    
    df['persistence_days'] = persistence_count
    df['direction_persistence'] = 0.0
    
    for i in range(len(df)):
        if df['persistence_days'].iloc[i] > 0:
            days = min(df['persistence_days'].iloc[i], 10)  # Cap at 10 days
            decay_sum = sum(0.8 ** j for j in range(days))
            df.loc[df.index[i], 'direction_persistence'] = decay_sum
    
    # Magnitude consistency
    df['trend_3d_std'] = df['trend_3d'].rolling(window=6).std()
    df['magnitude_consistency'] = 1 / (df['trend_3d_std'] + 0.001)
    
    # Combined persistence metric
    df['combined_persistence'] = (df['direction_persistence'] * 
                                 df['magnitude_consistency'] * 
                                 np.abs(df['trend_8d']))
    
    # Intraday Range Consistency Analysis
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['closing_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['closing_position'] = df['closing_position'].replace([np.inf, -np.inf], 0.5)
    df['closing_position'] = df['closing_position'].fillna(0.5)
    
    # Range-trend consistency
    df['range_consistency'] = np.abs(df['closing_position'] - 0.5) * 2
    
    # Final Alpha Construction
    # Base trend signal with consistency multiplier
    df['base_trend'] = df['trend_8d'] * np.where(df['trend_consistency'] > 0, 1.0, 0.5)
    
    # Volume-enhanced signal
    df['volume_enhanced'] = df['base_trend'] * df['volume_confidence']
    
    # Persistence-amplified signal
    df['persistence_amplified'] = df['volume_enhanced'] * df['combined_persistence']
    
    # Range-optimized alpha factor
    df['alpha_factor'] = (df['persistence_amplified'] * df['range_consistency']) / (df['daily_range'] + 0.001)
    
    # Clean up and return
    alpha_series = df['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
