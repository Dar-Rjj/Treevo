import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Momentum-Volume Divergence Convergence
    # Price and Volume Rate of Change (5-day lookback)
    data['price_roc'] = data['close'] / data['close'].shift(5) - 1
    data['volume_roc'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Divergence detection
    data['positive_divergence'] = ((data['price_roc'] > 0) & (data['volume_roc'] < 0)).astype(int)
    data['negative_divergence'] = ((data['price_roc'] < 0) & (data['volume_roc'] > 0)).astype(int)
    
    # Divergence duration and persistence
    data['divergence_duration'] = 0
    current_duration = 0
    for i in range(len(data)):
        if data['positive_divergence'].iloc[i] == 1 or data['negative_divergence'].iloc[i] == 1:
            current_duration += 1
        else:
            current_duration = 0
        data['divergence_duration'].iloc[i] = current_duration
    
    # Convergence acceleration
    data['divergence_change'] = data['positive_divergence'].diff() + data['negative_divergence'].diff()
    
    # Bid-Ask Spread Momentum
    data['relative_spread'] = (data['high'] - data['low']) / data['close']
    data['spread_roc'] = data['relative_spread'].pct_change()
    data['spread_volatility'] = data['relative_spread'].rolling(window=10, min_periods=5).std()
    
    # Spread compression/expansion signals
    spread_rolling = data['relative_spread'].rolling(window=20, min_periods=10)
    data['spread_percentile'] = spread_rolling.apply(lambda x: (x.iloc[-1] > x.quantile(0.8)) if len(x.dropna()) >= 10 else np.nan, raw=False)
    data['spread_compression'] = (data['relative_spread'] < data['relative_spread'].rolling(window=20, min_periods=10).quantile(0.2)).astype(int)
    
    # Volume-Weighted Price Acceleration
    data['price_change'] = data['close'].pct_change()
    data['vw_return'] = (data['price_change'] * data['volume']).rolling(window=5, min_periods=3).sum() / data['volume'].rolling(window=5, min_periods=3).sum()
    data['acceleration'] = data['vw_return'].diff()
    data['jerk'] = data['acceleration'].diff()
    
    # Acceleration regime detection
    data['positive_acceleration'] = (data['acceleration'] > 0).astype(int)
    data['acceleration_volume_corr'] = data['acceleration'].rolling(window=10, min_periods=5).corr(data['volume'].pct_change())
    
    # Price Range Efficiency Ratio
    data['net_movement'] = abs(data['close'] - data['close'].shift(1))
    data['total_oscillation'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).sum()
    data['efficiency_ratio'] = data['net_movement'].rolling(window=5, min_periods=3).sum() / data['total_oscillation']
    
    # Efficiency regime classification
    data['high_efficiency'] = (data['efficiency_ratio'] > data['efficiency_ratio'].rolling(window=20, min_periods=10).quantile(0.7)).astype(int)
    data['low_efficiency'] = (data['efficiency_ratio'] < data['efficiency_ratio'].rolling(window=20, min_periods=10).quantile(0.3)).astype(int)
    
    # Order Flow Imbalance Persistence
    data['buy_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'])).fillna(0.5)
    data['sell_pressure'] = 1 - data['buy_pressure']
    
    # Cumulative imbalance
    data['cumulative_imbalance'] = (data['buy_pressure'] - data['sell_pressure']).rolling(window=10, min_periods=5).sum()
    
    # Imbalance persistence
    data['imbalance_direction'] = np.sign(data['cumulative_imbalance'])
    data['imbalance_persistence'] = 0
    current_persistence = 0
    for i in range(len(data)):
        if i > 0 and data['imbalance_direction'].iloc[i] == data['imbalance_direction'].iloc[i-1]:
            current_persistence += 1
        else:
            current_persistence = 1
        data['imbalance_persistence'].iloc[i] = current_persistence
    
    # Imbalance acceleration and volatility
    data['imbalance_acceleration'] = data['cumulative_imbalance'].diff()
    data['imbalance_volatility'] = data['cumulative_imbalance'].rolling(window=10, min_periods=5).std()
    
    # Final alpha factor - weighted combination of components
    alpha = (
        0.25 * data['divergence_duration'] * np.sign(data['price_roc']) +
        0.20 * data['spread_compression'] * data['acceleration'] +
        0.25 * data['efficiency_ratio'] * data['imbalance_persistence'] +
        0.15 * data['vw_return'] * data['imbalance_acceleration'] +
        0.15 * data['jerk'] * data['divergence_change']
    )
    
    return alpha
