import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Regime Adjusted Momentum Divergence
    # Dynamic Volatility Context
    data['range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['range_20d'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['vol_ratio'] = data['range_5d'] / data['range_20d']
    data['vol_regime'] = np.where(data['vol_ratio'] > 1.2, 1.5, 
                                 np.where(data['vol_ratio'] < 0.8, 0.7, 1.0))
    
    # Intraday Momentum Component
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['gap_momentum'] = (data['close'] - data['open']) / data['open']
    data['intraday_signal'] = data['gap_momentum'] * data['intraday_range'] * data['vol_regime']
    
    # Volume-Price Divergence Detection
    data['price_return_5d'] = data['close'].pct_change(5)
    data['volume_momentum'] = data['volume'].pct_change(5)
    data['volume_price_corr'] = data['price_return_5d'].rolling(window=10).corr(data['volume_momentum'])
    data['divergence_strength'] = np.where(
        data['volume_price_corr'] < -0.3, 
        data['price_return_5d'] * data['vol_regime'],
        data['price_return_5d'] * 0.5
    )
    
    # Liquidity-Filtered Range Breakout
    data['avg_range_5d'] = (data['high'] - data['low']).rolling(window=5).mean()
    data['breakout_threshold'] = data['avg_range_5d'] * data['vol_regime']
    
    # Early Session Momentum (using first hour proxy - first 25% of range)
    data['first_hour_move'] = (data['high'].shift(1) - data['open'].shift(1)) / data['open'].shift(1)
    data['momentum_strength'] = np.abs(data['first_hour_move'])
    
    # Liquidity metrics
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['breakout_quality'] = (data['momentum_strength'] * 
                               (2 - data['spread_proxy']) * 
                               np.sqrt(data['volume_concentration']))
    
    # Gap Filling Probability
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_magnitude'] = np.abs(data['overnight_gap'])
    
    # Intraday fade detection
    data['early_volume_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['fade_progression'] = (data['close'] - data['open']) / data['overnight_gap'].replace(0, 0.001)
    data['fade_strength'] = data['fade_progression'] * data['early_volume_ratio']
    
    # Historical gap behavior
    data['gap_fill_ratio'] = (data['overnight_gap'].shift(1).abs() - 
                             (data['high'].shift(1) - data['low'].shift(1)).abs() / data['close'].shift(1))
    data['gap_probability'] = data['fade_strength'] * data['gap_fill_ratio'].rolling(window=10).mean()
    
    # Decay-Weighted Trend Strength
    # Multi-period momentum with exponential decay
    periods = [1, 3, 5, 10]
    weights = [0.4, 0.3, 0.2, 0.1]  # Higher weight to recent periods
    momentum_components = []
    
    for period, weight in zip(periods, weights):
        momentum = data['close'].pct_change(period) * weight
        momentum_components.append(momentum)
    
    data['decay_momentum'] = sum(momentum_components)
    
    # Volume-Price Alignment
    data['price_slope'] = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    data['volume_slope'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    data['trend_alignment'] = np.sign(data['price_slope']) * np.sign(data['volume_slope'])
    
    # Final composite factor
    data['composite_factor'] = (
        data['intraday_signal'] * 0.25 +
        data['divergence_strength'] * 0.20 +
        data['breakout_quality'] * 0.20 +
        data['gap_probability'] * 0.15 +
        data['decay_momentum'] * 0.10 +
        data['trend_alignment'] * 0.10
    )
    
    # Normalize by volatility
    data['rolling_vol'] = data['close'].pct_change().rolling(window=20).std()
    data['final_factor'] = data['composite_factor'] / data['rolling_vol'].replace(0, 0.001)
    
    return data['final_factor']
