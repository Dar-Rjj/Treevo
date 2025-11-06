import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum factor combining multi-timeframe momentum
    with volume confirmation and divergence signals.
    """
    # Multi-timeframe Momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_ratio'] = (df['close'] / df['close'].shift(5)) / (df['close'] / df['close'].shift(10))
    
    # Volume Confirmation
    df['volume_trend'] = df['volume'] / df['volume'].shift(5)
    df['volume_momentum'] = (df['volume'] / df['volume'].shift(5)) / (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume persistence (count of days with volume > previous day's volume over last 5 days)
    df['volume_persistence'] = 0
    for i in range(len(df)):
        if i >= 5:
            window = df['volume'].iloc[i-5:i+1]
            persistence_count = sum(window.iloc[j] > window.iloc[j-1] for j in range(1, len(window)))
            df.loc[df.index[i], 'volume_persistence'] = persistence_count
    
    # Divergence Signal
    df['bullish_divergence'] = (df['momentum_5d'] > df['momentum_5d'].shift(1)) & (df['volume_trend'] < 1)
    df['bearish_divergence'] = (df['momentum_5d'] < df['momentum_5d'].shift(1)) & (df['volume_trend'] > 1)
    
    # Signal strength
    df['signal_strength'] = df['momentum_5d'] * (1 / df['volume_trend'])
    
    # High-Low Range Efficiency
    df['daily_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Gap-adjusted efficiency
    df['gap_high'] = np.maximum(df['high'], df['close'].shift(1))
    df['gap_low'] = np.minimum(df['low'], df['close'].shift(1))
    df['gap_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['gap_high'] - df['gap_low'])
    
    # 3-day cumulative efficiency
    df['price_change_abs'] = abs(df['close'] - df['close'].shift(1))
    df['daily_range'] = df['high'] - df['low']
    df['cumulative_efficiency_3d'] = (
        df['price_change_abs'].rolling(window=3).sum() / 
        df['daily_range'].rolling(window=3).sum()
    )
    
    # Efficiency persistence
    df['efficiency_5d_avg'] = df['daily_efficiency'].rolling(window=5).mean()
    df['efficiency_trend'] = df['daily_efficiency'] / df['efficiency_5d_avg']
    
    # Efficiency consistency (count efficiency > threshold over last 5 days)
    threshold = df['daily_efficiency'].rolling(window=20).quantile(0.6)
    df['efficiency_consistency'] = (
        (df['daily_efficiency'] > threshold).rolling(window=5).sum()
    )
    
    # Volume-Scaled Extreme Reversal
    df['price_extremity'] = (
        (df['close'] - df['close'].rolling(window=3).mean()) / 
        (df['close'].rolling(window=3).max() - df['close'].rolling(window=3).min())
    )
    df['volume_extremity'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['combined_extreme_score'] = df['price_extremity'] * df['volume_extremity']
    
    # Amount Flow Momentum
    df['net_flow'] = np.sign(df['close'] - df['close'].shift(1)) * df['amount']
    df['flow_momentum'] = (
        df['net_flow'].rolling(window=3).sum() / 
        df['amount'].rolling(window=3).sum()
    )
    
    # Flow consistency (count same direction flow over last 5 days)
    df['flow_direction'] = np.sign(df['net_flow'])
    df['flow_consistency'] = (
        (df['flow_direction'] == df['flow_direction'].shift(1)).rolling(window=5).sum()
    )
    
    # Volatility-Adaptive Volume Patterns
    df['volatility_5d'] = df['close'].rolling(window=5).std()
    df['volatility_10d'] = df['close'].rolling(window=10).std()
    df['relative_volatility'] = df['volatility_5d'] / df['volatility_10d'].shift(5)
    
    # Volume spike detection
    df['volume_median_10d'] = df['volume'].rolling(window=10).median()
    df['volume_spike'] = df['volume'] / df['volume_median_10d']
    
    # Final factor combining key components
    factor = (
        df['momentum_ratio'] * 0.3 +
        df['signal_strength'] * 0.2 +
        df['cumulative_efficiency_3d'] * 0.15 +
        df['combined_extreme_score'] * 0.15 +
        df['flow_momentum'] * 0.1 +
        df['volume_spike'] * 0.05 +
        df['relative_volatility'] * 0.05
    )
    
    return factor
