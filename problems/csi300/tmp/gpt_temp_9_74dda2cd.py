import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Quality with Volume-Verified Range Efficiency
    Combines momentum quality assessment across multiple timeframes with volume-verified range efficiency analysis
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Quality Assessment
    # Short-Term Momentum (3-day)
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_1d'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_acceleration'] = data['momentum_3d'] - data['momentum_1d']
    
    # Medium-Term Momentum (10-day)
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_stability'] = np.where(data['momentum_3d'] != 0, 
                                        data['momentum_10d'] / data['momentum_3d'], 0)
    
    # Long-Term Momentum (20-day)
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['long_term_trend'] = np.sign(data['momentum_20d'])
    
    # Momentum Quality Analysis
    # Momentum Purity (consistency of daily returns)
    data['daily_returns'] = data['close'].pct_change()
    data['momentum_purity'] = data['daily_returns'].rolling(window=5).apply(
        lambda x: np.mean(np.sign(x) == np.sign(np.mean(x))) if len(x) == 5 else 0
    )
    
    # Momentum Smoothness (low volatility in momentum)
    data['momentum_smoothness'] = 1 / (1 + data['momentum_3d'].rolling(window=5).std())
    
    # Momentum Persistence across timeframes
    data['momentum_alignment'] = (np.sign(data['momentum_3d']) * np.sign(data['momentum_10d']) * 
                                np.sign(data['momentum_20d']))
    data['momentum_persistence'] = data['momentum_alignment'].rolling(window=5).mean()
    
    # Volume-Verified Range Efficiency Analysis
    # Range Efficiency Measurement
    data['daily_range'] = data['high'] - data['low']
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['range_efficiency'] = abs(data['close'] - data['close'].shift(1)) / data['true_range']
    data['range_efficiency'] = data['range_efficiency'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Smart Money Volume Analysis
    data['avg_trade_size'] = data['amount'] / data['volume']
    data['large_trade_concentration'] = data['avg_trade_size'].rolling(window=5).apply(
        lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0
    )
    
    # Volume-Range Confirmation
    data['volume_range_ratio'] = data['volume'] / data['daily_range']
    data['volume_range_ratio'] = data['volume_range_ratio'].replace([np.inf, -np.inf], 0).fillna(0)
    
    # Volume clustering at range extremes
    data['high_volume_cluster'] = (data['volume'] > data['volume'].rolling(window=10).quantile(0.7)) & \
                                (abs(data['close'] - data['high']) / data['daily_range'] < 0.2)
    data['low_volume_cluster'] = (data['volume'] > data['volume'].rolling(window=10).quantile(0.7)) & \
                               (abs(data['close'] - data['low']) / data['daily_range'] < 0.2)
    
    # Generate Combined Signals
    # High quality momentum + efficient range + smart money confirmation
    momentum_quality = (data['momentum_purity'] + data['momentum_smoothness'] + 
                       data['momentum_persistence']) / 3
    
    range_efficiency_score = data['range_efficiency'].rolling(window=5).mean()
    
    smart_money_confirmation = (data['large_trade_concentration'].rolling(window=5).mean() * 
                              np.sign(data['momentum_3d']))
    
    # Final factor calculation
    factor = (momentum_quality * 0.4 + 
             range_efficiency_score * 0.3 + 
             smart_money_confirmation * 0.3)
    
    # Handle any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
