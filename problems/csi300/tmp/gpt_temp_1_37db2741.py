import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Weighted Momentum-Range Divergence
    # Dual-Period Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Range Dynamics
    data['range_5d'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / data['close'].shift(5)
    data['range_10d'] = (data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()) / data['close'].shift(10)
    
    # True Range volatility (20-day average)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_20d'] = data['true_range'].rolling(window=20).mean()
    
    # Momentum-Range Divergence
    data['momentum_range_div'] = (data['momentum_5d'] - data['momentum_10d']) - (data['range_5d'] - data['range_10d'])
    # Volatility weighting
    vol_rank = data['volatility_20d'].rolling(window=50).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    data['vol_weighted_div'] = data['momentum_range_div'] * (0.5 + vol_rank)
    
    # Volume-Confirmed Price Acceleration with Persistence
    # Multi-Timeframe Acceleration
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['acceleration'] = data['price_change'] - data['price_change'].shift(1)
    
    # Acceleration persistence over 3-day window
    data['accel_persistence'] = data['acceleration'].rolling(window=3).apply(
        lambda x: 1 if all(x > 0) else (-1 if all(x < 0) else 0), raw=False
    )
    
    # Volume momentum
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Volume-confirmed acceleration
    data['volume_accel_signal'] = np.where(
        (data['acceleration'] > 0) & (data['volume_momentum'] > 0) & (data['accel_persistence'] > 0), 1,
        np.where(
            (data['acceleration'] < 0) & (data['volume_momentum'] > 0) & (data['accel_persistence'] < 0), -1, 0
        )
    )
    
    # Intraday Pressure Efficiency with Range Confirmation
    # Intraday Pressure Components
    data['bullish_pressure'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['bearish_pressure'] = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    data['net_pressure'] = data['bullish_pressure'] - data['bearish_pressure']
    
    # Range Efficiency Assessment
    data['range_utilization'] = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['pressure_efficiency'] = data['net_pressure'] * data['range_utilization']
    
    # Volume-Weighted Persistence
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['pressure_signal'] = data['pressure_efficiency'] * data['volume_ratio']
    data['pressure_persistence'] = data['pressure_signal'].rolling(window=3).mean()
    
    # Gap Momentum with Volatility Adjustment
    # Gap Strength and Persistence
    data['daily_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap persistence over 3 days
    data['gap_direction'] = np.sign(data['daily_gap'])
    data['gap_persistence'] = data['gap_direction'].rolling(window=3).apply(
        lambda x: 1 if len(set(x)) == 1 and x.iloc[-1] > 0 else (-1 if len(set(x)) == 1 and x.iloc[-1] < 0 else 0), 
        raw=False
    )
    
    # Volatility-Adjusted Gap Signal
    data['volatility_5d'] = data['true_range'].rolling(window=5).mean()
    vol_adj = 1.5 - abs(data['volatility_5d'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std(), raw=False
    ))
    data['vol_adj_gap'] = data['daily_gap'] * np.clip(vol_adj, 0.5, 2.0)
    
    # Range Follow-Through Confirmation
    data['gap_follow_through'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['gap_efficiency'] = data['vol_adj_gap'] * data['gap_follow_through'] * data['range_utilization']
    
    # Multi-Timeframe Volume-Price-Range Alignment
    # Short-term components (1-3 days)
    data['short_term_momentum'] = data['close'] / data['close'].shift(3) - 1
    data['short_term_accel'] = data['acceleration'].rolling(window=3).mean()
    data['short_term_volume'] = data['volume'] / data['volume'].shift(3) - 1
    
    # Medium-term components (5-10 days)
    data['medium_term_momentum'] = data['momentum_5d']
    data['medium_term_range'] = data['range_5d']
    data['medium_term_gap'] = data['gap_efficiency'].rolling(window=5).mean()
    
    # Composite Alignment Scoring
    alignment_components = [
        np.sign(data['short_term_momentum']) == np.sign(data['medium_term_momentum']),
        np.sign(data['short_term_accel']) == np.sign(data['medium_term_momentum']),
        np.sign(data['short_term_volume']) == np.sign(data['short_term_momentum']),
        abs(data['pressure_persistence']) > 0.1,
        abs(data['gap_efficiency']) > 0.01
    ]
    
    data['alignment_score'] = sum(components.astype(int) for components in alignment_components)
    data['weighted_alignment'] = data['alignment_score'] * data['volume_ratio'] * data['range_utilization']
    
    # Adaptive Pressure Release with Volume Clustering
    # Accumulated Pressure
    data['daily_pressure'] = (2 * data['close'] - data['low'] - data['high']) / (data['high'] - data['low'] + 1e-8)
    data['accumulated_pressure'] = data['daily_pressure'].rolling(window=5).sum()
    
    # Volume Cluster Detection
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    data['volume_cluster'] = data['volume_zscore'] > 1.5
    
    # Pressure release detection
    data['pressure_change'] = data['accumulated_pressure'].diff()
    data['pressure_release'] = (data['pressure_change'] < -0.1) & data['volume_cluster']
    
    # Adaptive Signal Generation
    data['post_release_move'] = data['close'].shift(-1) / data['close'] - 1  # Note: This is for signal logic, not for forward-looking
    data['pressure_signal_adaptive'] = np.where(
        data['pressure_release'] & (data['post_release_move'].shift(1) > 0) & (data['daily_pressure'] > 0), 1,
        np.where(
            data['pressure_release'] & (data['post_release_move'].shift(1) < 0) & (data['daily_pressure'] < 0), -1, 0
        )
    )
    
    # Final composite factor
    components = [
        data['vol_weighted_div'],
        data['volume_accel_signal'],
        data['pressure_persistence'],
        data['gap_efficiency'],
        data['weighted_alignment'],
        data['pressure_signal_adaptive']
    ]
    
    # Normalize and combine components
    normalized_components = []
    for component in components:
        if component.notna().any():
            normalized = (component - component.rolling(window=50).mean()) / component.rolling(window=50).std()
            normalized_components.append(normalized.fillna(0))
    
    # Equal weighted combination
    final_factor = sum(normalized_components) / len(normalized_components)
    
    return final_factor
