import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Momentum Divergence
    df['price_momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['volume_trend'] = df['volume'] / df['volume'].shift(5)
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(5)) / (df['volume'].shift(5) / df['volume'].shift(10))
    df['momentum_divergence'] = np.where(df['price_momentum_5d'] > df['volume_trend'], 1, -1)
    
    # Range Efficiency Factor
    df['daily_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['cumulative_range_3d'] = df['high'].rolling(window=3).max() - df['low'].rolling(window=3).min()
    df['efficiency_3d'] = abs(df['close'] - df['close'].shift(3)) / df['cumulative_range_3d']
    df['efficiency_trend'] = df['daily_efficiency'] / df['daily_efficiency'].shift(3)
    df['efficiency_persistence'] = df['daily_efficiency'].rolling(window=5).apply(lambda x: (x > 0.5).sum())
    
    # Volume-Confirmed Reversal
    df['price_deviation'] = (df['close'] - df['close'].shift(3)) / (df['high'].rolling(window=4).max() - df['low'].rolling(window=4).min())
    df['volume_spike'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['reversal_strength'] = df['price_deviation'] * df['volume_spike']
    
    # Amount Flow Direction
    df['up_amount'] = np.where(df['close'] > df['close'].shift(1), df['amount'], 0)
    df['down_amount'] = np.where(df['close'] < df['close'].shift(1), df['amount'], 0)
    df['net_flow'] = df['up_amount'] - df['down_amount']
    
    def flow_consistency(x):
        if len(x) < 2:
            return 0
        return sum(np.sign(x.iloc[i]) == np.sign(x.iloc[i-1]) for i in range(1, len(x)))
    
    df['flow_consistency'] = df['net_flow'].rolling(window=5).apply(flow_consistency, raw=False)
    df['flow_acceleration'] = df['net_flow'] / df['net_flow'].shift(3)
    
    # Volatility-Volume Regime
    df['volatility_range'] = (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()) / df['close'].shift(10)
    df['volatility_ratio'] = df['close'].rolling(window=5).std() / df['close'].shift(5).rolling(window=5).std()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=10).mean()
    
    def volume_persistence(x):
        mean_vol = x.rolling(window=10).mean().iloc[-1] if len(x) >= 10 else x.mean()
        return (x > mean_vol).sum()
    
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(volume_persistence, raw=False)
    
    # Combine factors with weights
    factor = (
        0.25 * df['momentum_divergence'] +
        0.20 * df['efficiency_3d'] +
        0.15 * df['reversal_strength'] +
        0.20 * df['net_flow'] / df['net_flow'].rolling(window=20).std() +
        0.20 * np.where(
            (df['volatility_range'] > df['volatility_range'].quantile(0.7)) & 
            (df['volume_ratio'] > 1.2), 
            df['price_momentum_5d'], 
            -df['price_momentum_5d']
        )
    )
    
    return factor
