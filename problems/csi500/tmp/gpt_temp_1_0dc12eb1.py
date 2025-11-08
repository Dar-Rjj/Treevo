import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate true range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Breakout Detection
    df['20d_high'] = df['high'].rolling(window=20, min_periods=10).max()
    df['20d_low'] = df['low'].rolling(window=20, min_periods=10).min()
    
    # Breakout signals and distances
    df['breakout_high'] = (df['close'] > df['20d_high']).astype(int)
    df['breakout_low'] = (df['close'] < df['20d_low']).astype(int)
    df['breakout_distance'] = np.where(
        df['breakout_high'] == 1, 
        (df['close'] - df['20d_high']) / df['20d_high'],
        np.where(
            df['breakout_low'] == 1,
            (df['close'] - df['20d_low']) / df['20d_low'],
            0
        )
    )
    
    # Breakout Efficiency (5-day window after breakout)
    df['net_movement'] = df['close'] - df['close'].shift(5)
    df['total_volatility'] = df['true_range'].rolling(window=5, min_periods=3).sum()
    df['breakout_efficiency'] = df['net_movement'] / df['total_volatility']
    df['breakout_efficiency'] = df['breakout_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Momentum Acceleration
    # Price Efficiency Momentum (5-day)
    df['price_change'] = df['close'].diff()
    df['abs_price_change'] = abs(df['price_change'])
    df['net_5d_change'] = df['close'] - df['close'].shift(5)
    df['total_5d_abs_change'] = df['abs_price_change'].rolling(window=5, min_periods=3).sum()
    df['price_efficiency'] = df['net_5d_change'] / df['total_5d_abs_change']
    df['price_efficiency'] = df['price_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Range Efficiency Momentum
    df['range_efficiency_5d'] = df['net_5d_change'] / df['true_range'].rolling(window=5, min_periods=3).sum()
    df['range_efficiency_10d'] = (df['close'] - df['close'].shift(10)) / df['true_range'].rolling(window=10, min_periods=5).sum()
    df['range_efficiency_momentum'] = df['range_efficiency_5d'] - df['range_efficiency_10d']
    df['range_efficiency_momentum'] = df['range_efficiency_momentum'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume Confirmation
    # Volume-Range Correlation Break
    df['volume_range_corr_15d'] = df['volume'].rolling(window=15, min_periods=8).corr(df['true_range'])
    df['volume_range_corr_30d_avg'] = df['volume_range_corr_15d'].rolling(window=30, min_periods=15).mean()
    df['correlation_break'] = df['volume_range_corr_15d'] - df['volume_range_corr_30d_avg']
    
    # Breakout Strength Persistence
    df['strong_breakout'] = ((df['breakout_distance'].abs() > df['breakout_distance'].abs().rolling(window=20, min_periods=10).mean()) & 
                            (df['breakout_efficiency'].abs() > df['breakout_efficiency'].abs().rolling(window=20, min_periods=10).mean())).astype(int)
    df['breakout_direction'] = np.sign(df['breakout_distance'])
    df['persistence_count'] = 0
    
    for i in range(1, len(df)):
        if df['strong_breakout'].iloc[i] == 1:
            if df['breakout_direction'].iloc[i] == df['breakout_direction'].iloc[i-1]:
                df.loc[df.index[i], 'persistence_count'] = df['persistence_count'].iloc[i-1] + df['breakout_direction'].iloc[i]
            else:
                df.loc[df.index[i], 'persistence_count'] = df['breakout_direction'].iloc[i]
        else:
            df.loc[df.index[i], 'persistence_count'] = 0
    
    # Signal Integration
    df['base_signal'] = df['breakout_efficiency'] * (1 + df['range_efficiency_momentum'])
    df['volume_confirmation'] = 1 + (df['correlation_break'] * df['persistence_count'].abs())
    df['final_signal'] = df['base_signal'] * df['volume_confirmation'] * (1 + 0.1 * df['persistence_count'])
    
    # Clean up intermediate columns
    result = df['final_signal'].copy()
    return result
