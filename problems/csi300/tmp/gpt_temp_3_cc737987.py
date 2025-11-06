import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Frequency Price Trends
    # Short-term trend (1-3 days) - High-Low range momentum
    data['high_low_range'] = data['high'] - data['low']
    data['range_momentum_1d'] = data['high_low_range'] / data['high_low_range'].shift(1) - 1
    data['range_momentum_3d'] = data['high_low_range'] / data['high_low_range'].shift(3) - 1
    data['short_term_trend'] = (data['range_momentum_1d'] + data['range_momentum_3d']) / 2
    
    # Medium-term trend (5-10 days) - Close price acceleration
    data['close_roc_5d'] = data['close'] / data['close'].shift(5) - 1
    data['close_roc_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_acceleration'] = data['close_roc_5d'] - data['close_roc_10d']
    
    # Long-term trend (15-20 days) - Open-Close consistency
    data['oc_ratio'] = (data['close'] - data['open']) / data['open']
    data['oc_consistency_15d'] = data['oc_ratio'].rolling(window=15).apply(
        lambda x: np.mean(np.sign(x) == np.sign(np.mean(x))) if len(x) == 15 else np.nan
    )
    data['oc_consistency_20d'] = data['oc_ratio'].rolling(window=20).apply(
        lambda x: np.mean(np.sign(x) == np.sign(np.mean(x))) if len(x) == 20 else np.nan
    )
    data['long_term_trend'] = (data['oc_consistency_15d'] + data['oc_consistency_20d']) / 2
    
    # Liquidity Conditions
    # Volume-Price Efficiency
    data['price_change'] = data['close'] - data['close'].shift(1)
    data['abs_price_change'] = abs(data['price_change'])
    data['volume_efficiency'] = data['abs_price_change'] / (data['volume'] + 1e-8)
    data['expected_volume_impact'] = data['volume_efficiency'].rolling(window=10).mean()
    data['volume_price_efficiency'] = data['expected_volume_impact'] / (data['volume_efficiency'] + 1e-8)
    
    # Bid-Ask Spread Proxy
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high_low_range'] + 1e-8)
    data['range_efficiency_ratio'] = data['range_efficiency'] / data['range_efficiency'].rolling(window=10).mean()
    
    # Order Flow Imbalance
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    data['momentum_persistence'] = data['intraday_momentum'].rolling(window=5).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan
    )
    
    # Momentum Convergence Patterns
    # Multi-Frequency Alignment
    data['trend_alignment'] = (
        np.sign(data['short_term_trend']) * np.sign(data['price_acceleration']) * 
        np.sign(data['long_term_trend'])
    )
    data['convergence_strength'] = (
        abs(data['short_term_trend']) + abs(data['price_acceleration']) + abs(data['long_term_trend'])
    ) * data['trend_alignment']
    
    # Liquidity-Momentum Interaction
    data['liquidity_quality'] = (
        data['volume_price_efficiency'] * data['range_efficiency_ratio'] * data['momentum_persistence']
    )
    data['liquidity_adjusted_momentum'] = data['convergence_strength'] * data['liquidity_quality']
    
    # Dynamic Signal Weighting
    # Recent Market Conditions
    data['price_volatility'] = data['close'].pct_change().rolling(window=10).std()
    data['volatility_adjustment'] = 1 / (1 + data['price_volatility'])
    
    # Volume Confirmation
    data['volume_trend'] = data['volume'].rolling(window=5).apply(
        lambda x: 1 if x.is_monotonic_increasing else (-1 if x.is_monotonic_decreasing else 0)
    )
    data['volume_confirmation'] = data['volume_trend'] * np.sign(data['liquidity_adjusted_momentum'])
    
    # Final Alpha Factor
    data['alpha_factor'] = (
        data['liquidity_adjusted_momentum'] * 
        data['volatility_adjustment'] * 
        (1 + 0.2 * data['volume_confirmation'])
    )
    
    # Clean up and return
    result = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
