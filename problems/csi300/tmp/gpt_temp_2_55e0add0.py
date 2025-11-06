import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Price-Volume Efficiency Momentum
    # Price Efficiency Ratio
    df['close_5d_diff'] = df['close'] - df['close'].shift(5)
    df['abs_daily_change'] = abs(df['close'] - df['close'].shift(1))
    df['sum_abs_5d'] = df['abs_daily_change'].rolling(window=5, min_periods=5).sum()
    df['efficiency_ratio'] = df['close_5d_diff'] / df['sum_abs_5d']
    
    # Volume-Weighted Momentum
    df['momentum_3d'] = df['close'] - df['close'].shift(3)
    df['momentum_6d'] = df['close'] - df['close'].shift(6)
    df['avg_volume_3d'] = (df['volume'] + df['volume'].shift(1) + df['volume'].shift(2)) / 3
    df['weighted_momentum'] = df['momentum_3d'] * df['momentum_6d'] / df['avg_volume_3d']
    
    df['factor1'] = df['efficiency_ratio'] * df['weighted_momentum']
    
    # Range-Based Volume Divergence
    # Range Efficiency
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['range_efficiency'] = (df['close'] - df['close'].shift(1)) / df['true_range']
    
    # Volume Acceleration
    df['volume_acceleration'] = df['volume'] / ((df['volume'] + df['volume'].shift(1) + df['volume'].shift(2)) / 3)
    df['volume_comparison'] = df['volume'] / df['volume'].shift(3)
    
    df['factor2'] = df['range_efficiency'] * df['volume_acceleration'] * df['volume_comparison']
    
    # Intraday Persistence with Liquidity
    # Close Position Strength
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['persistence_count'] = 0
    for i in range(1, len(df)):
        if df['close_position'].iloc[i] > 0.7:
            if df['close_position'].iloc[i-1] > 0.7:
                df['persistence_count'].iloc[i] = df['persistence_count'].iloc[i-1] + 1
            else:
                df['persistence_count'].iloc[i] = 1
        else:
            df['persistence_count'].iloc[i] = 0
    
    # Liquidity Adjustment
    df['liquidity_ratio'] = df['amount'] / ((df['amount'] + df['amount'].shift(1) + df['amount'].shift(2)) / 3)
    
    df['factor3'] = df['persistence_count'] * df['liquidity_ratio']
    
    # Volatility-Adjusted Flow Momentum
    # Order Flow Momentum
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['flow_t'] = df['typical_price'] * df['volume'] * np.sign(df['close'] - df['close'].shift(1))
    df['flow_t_3'] = df['typical_price'].shift(3) * df['volume'].shift(3) * np.sign(df['close'].shift(3) - df['close'].shift(4))
    df['flow_momentum'] = df['flow_t'] - df['flow_t_3']
    
    # Volatility Adjustment
    df['volatility'] = (df['high'] - df['low']) / df['typical_price']
    
    df['factor4'] = df['flow_momentum'] / (1 + df['volatility'])
    
    # Volume-Cluster Reversal Acceleration
    # Reversal Pattern
    df['reversal'] = (df['close'] - df['close'].shift(1)) * (df['close'].shift(1) - df['close'].shift(2))
    df['reversal_acceleration'] = df['reversal'] - df['reversal'].shift(1)
    
    # Volume Cluster Weight
    df['avg_volume_3d_cluster'] = (df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3)) / 3
    df['cluster_weight'] = np.where(df['volume'] > df['avg_volume_3d_cluster'], 1.0, 0.5)
    
    df['factor5'] = df['reversal_acceleration'] * df['cluster_weight']
    
    # Combine all factors
    df['final_factor'] = (
        df['factor1'].fillna(0) + 
        df['factor2'].fillna(0) + 
        df['factor3'].fillna(0) + 
        df['factor4'].fillna(0) + 
        df['factor5'].fillna(0)
    )
    
    return df['final_factor']
