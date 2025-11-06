import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    
    # Volatility Regime Detection
    df['short_term_vol'] = df['returns'].rolling(window=5).std()
    df['medium_term_vol'] = df['returns'].rolling(window=20).std()
    df['vol_regime'] = np.where(df['short_term_vol'] > df['medium_term_vol'], 'high', 'low')
    
    # Price Momentum Acceleration
    df['short_term_momentum'] = df['close'] / df['close'].shift(5)
    df['medium_term_momentum'] = df['close'].shift(5) / df['close'].shift(10)
    df['acceleration_signal'] = df['short_term_momentum'] - df['medium_term_momentum']
    
    # Volume Dynamics Analysis
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(5)) - (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume Cluster Detection
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['high_volume'] = df['volume'] > 1.5 * df['volume_ma_20']
    
    # Find volume clusters
    cluster_start = []
    cluster_end = []
    in_cluster = False
    cluster_start_idx = -1
    
    for i in range(len(df)):
        if df['high_volume'].iloc[i] and not in_cluster:
            in_cluster = True
            cluster_start_idx = i
        elif not df['high_volume'].iloc[i] and in_cluster:
            in_cluster = False
            cluster_start.append(cluster_start_idx)
            cluster_end.append(i-1)
    
    # Calculate cluster momentum
    df['cluster_momentum'] = 0.0
    for start_idx, end_idx in zip(cluster_start, cluster_end):
        if end_idx + 3 < len(df):
            cluster_return = df['close'].iloc[end_idx] / df['close'].iloc[start_idx] - 1
            follow_return = df['close'].iloc[end_idx + 3] / df['close'].iloc[end_idx] - 1
            df.loc[df.index[end_idx], 'cluster_momentum'] = cluster_return * follow_return
    
    # Price Quality Assessment
    df['intraday_strength'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['gap_persistence'] = (df['close'] / df['open']) - (df['close'].shift(1) / df['open'].shift(1))
    
    # Range Efficiency
    df['range_efficiency'] = 0.0
    for i in range(len(df)):
        if i >= 5:
            five_day_return = df['close'].iloc[i] / df['close'].iloc[i-5] - 1
            volatility_sum = 0
            for j in range(5):
                daily_return = abs(df['close'].iloc[i-j] / df['close'].iloc[i-j-1] - 1)
                volatility_sum += daily_return
            if volatility_sum != 0:
                df.loc[df.index[i], 'range_efficiency'] = five_day_return / volatility_sum
    
    # Strength Persistence (consecutive days with intraday strength > 0.7)
    df['strength_persistence'] = 0
    current_streak = 0
    for i in range(len(df)):
        if df['intraday_strength'].iloc[i] > 0.7:
            current_streak += 1
        else:
            current_streak = 0
        df.loc[df.index[i], 'strength_persistence'] = current_streak
    
    # Regime-Adaptive Factor Construction
    df['momentum_breakout'] = df['acceleration_signal'] * (df['gap_persistence'] > 0) * df['volume_acceleration']
    df['cluster_confirmation'] = df['cluster_momentum'] * df['intraday_strength']
    df['efficient_momentum'] = df['acceleration_signal'] * df['range_efficiency']
    
    # Adaptive Signal Integration
    df['regime_factor'] = 0.0
    high_vol_mask = df['vol_regime'] == 'high'
    low_vol_mask = df['vol_regime'] == 'low'
    
    df.loc[high_vol_mask, 'regime_factor'] = (
        df.loc[high_vol_mask, 'momentum_breakout'] + 
        df.loc[high_vol_mask, 'cluster_confirmation']
    )
    
    df.loc[low_vol_mask, 'regime_factor'] = (
        df.loc[low_vol_mask, 'efficient_momentum'] + 
        df.loc[low_vol_mask, 'strength_persistence']
    )
    
    # Volume-Weighted Final Alpha
    df['volume_weight'] = (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2)) / 3
    df['final_signal'] = df['regime_factor'] * df['volume_weight']
    
    # Clean up intermediate columns
    result = df['final_signal'].copy()
    
    return result
