import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volume Efficiency Divergence factor
    Combines short, medium, and long-term volume efficiency measures
    with regime-adaptive dynamics and breakout analytics
    """
    
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Calculate basic price features
    data['returns'] = data['close'].pct_change()
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Short-term efficiency (3-day)
    data['short_return_per_volume'] = data['returns'].rolling(window=3).sum() / data['volume'].rolling(window=3).mean()
    data['short_volume_concentration'] = (data['volume'] * abs(data['returns'])).rolling(window=3).sum() / data['volume'].rolling(window=3).sum()
    
    # Medium-term efficiency (10-day)
    data['medium_cum_return'] = data['returns'].rolling(window=10).sum()
    data['medium_avg_volume'] = data['volume'].rolling(window=10).mean()
    data['medium_efficiency'] = data['medium_cum_return'] / data['medium_avg_volume']
    data['medium_range_utilization'] = data['price_range'].rolling(window=10).mean()
    
    # Long-term efficiency (20-day)
    data['long_efficiency'] = data['returns'].rolling(window=20).sum() / data['volume'].rolling(window=20).mean()
    data['long_efficiency_trend'] = data['long_efficiency'].diff(5)
    data['long_efficiency_acceleration'] = data['long_efficiency_trend'].diff(3)
    
    # Volatility regime classification
    data['volatility_20d'] = data['returns'].rolling(window=20).std()
    data['volatility_regime'] = np.where(data['volatility_20d'] > data['volatility_20d'].rolling(window=60).quantile(0.7), 
                                        'high', 
                                        np.where(data['volatility_20d'] < data['volatility_20d'].rolling(window=60).quantile(0.3), 
                                                'low', 'medium'))
    
    # Trending vs mean-reversion efficiency
    data['trend_strength'] = data['close'].rolling(window=10).apply(lambda x: (x[-1] - x[0]) / (x.max() - x.min() + 1e-8))
    data['trending_efficiency'] = data['medium_efficiency'] * np.where(data['trend_strength'] > 0.3, 1, 
                                                                      np.where(data['trend_strength'] < -0.3, -1, 0))
    
    # Multi-timeframe momentum alignment
    data['momentum_short'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_long'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_alignment'] = np.sign(data['momentum_short']) + np.sign(data['momentum_medium']) + np.sign(data['momentum_long'])
    
    # Advanced breakout analytics
    data['price_breakout'] = (data['close'] > data['high'].rolling(window=10).max().shift(1)).astype(int)
    data['volume_breakout'] = (data['volume'] > data['volume'].rolling(window=20).mean() * 1.5).astype(int)
    data['breakout_quality'] = data['price_breakout'] * data['volume_breakout'] * data['momentum_alignment']
    
    # Trend exhaustion signals
    data['exhaustion_signal'] = (abs(data['momentum_short']) > 0.08) & (data['volume'] > data['volume'].rolling(window=20).mean() * 2)
    
    # Multi-dimensional integration with dynamic weights
    # Volatility-adjusted weights
    high_vol_weight = np.where(data['volatility_regime'] == 'high', 0.3, 0.6)
    medium_vol_weight = np.where(data['volatility_regime'] == 'medium', 0.5, 0.3)
    low_vol_weight = np.where(data['volatility_regime'] == 'low', 0.2, 0.1)
    
    # Timeframe weights based on momentum alignment
    timeframe_weight = np.where(abs(data['momentum_alignment']) == 3, 0.4, 
                               np.where(abs(data['momentum_alignment']) == 2, 0.3, 0.2))
    
    # Efficiency divergence components
    short_efficiency_div = data['short_return_per_volume'] - data['short_return_per_volume'].rolling(window=10).mean()
    medium_efficiency_div = data['medium_efficiency'] - data['medium_efficiency'].rolling(window=20).mean()
    long_efficiency_div = data['long_efficiency'] - data['long_efficiency'].rolling(window=40).mean()
    
    # Final factor construction
    factor = (
        high_vol_weight * short_efficiency_div +
        medium_vol_weight * medium_efficiency_div +
        low_vol_weight * long_efficiency_div
    ) * timeframe_weight
    
    # Adjust for breakout quality
    factor = factor * (1 + 0.2 * data['breakout_quality'])
    
    # Adjust for trend exhaustion (reduce signal during exhaustion)
    factor = factor * np.where(data['exhaustion_signal'], 0.5, 1.0)
    
    # Normalize by volatility regime
    factor = factor / (data['volatility_20d'] + 1e-8)
    
    # Final smoothing and cleaning
    factor = factor.rolling(window=5).mean()
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    return factor
