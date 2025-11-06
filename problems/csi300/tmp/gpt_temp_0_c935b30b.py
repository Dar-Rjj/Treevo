import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Price Reversal Patterns with Liquidity Dynamics
    Combines multi-dimensional reversal signals with liquidity analysis
    """
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Dimensional Reversal Signal Detection
    # Extreme Price Movement Identification
    data['range'] = data['high'] - data['low']
    data['range_avg_4d'] = data['range'].rolling(window=4, min_periods=1).mean()
    data['range_expansion'] = data['range'] / data['range_avg_4d'].shift(1)
    
    # Close-to-Close momentum extremes
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['momentum_percentile'] = data['close_ret'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Opening gap reversal potential
    data['open_gap'] = (data['open'] / data['close'].shift(1) - 1)
    data['recent_volatility'] = data['close_ret'].rolling(window=10, min_periods=1).std()
    data['gap_normalized'] = data['open_gap'] / (data['recent_volatility'] + 1e-8)
    
    # Reversal Confirmation Framework
    # Price action reversal patterns
    data['body'] = abs(data['close'] - data['open'])
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    
    # Hammer pattern (long lower shadow, small body)
    hammer_condition = (
        (data['lower_shadow'] > 2 * data['body']) & 
        (data['upper_shadow'] < 0.3 * data['body']) &
        (data['close'] < data['open'])  # Bearish day
    )
    
    # Shooting star pattern (long upper shadow, small body)
    shooting_star_condition = (
        (data['upper_shadow'] > 2 * data['body']) & 
        (data['lower_shadow'] < 0.3 * data['body']) &
        (data['close'] > data['open'])  # Bullish day
    )
    
    # Engulfing pattern
    data['prev_body'] = abs(data['close'].shift(1) - data['open'].shift(1))
    bullish_engulfing = (
        (data['close'] > data['open']) & 
        (data['close'].shift(1) < data['open'].shift(1)) &
        (data['close'] > data['open'].shift(1)) & 
        (data['open'] < data['close'].shift(1))
    )
    bearish_engulfing = (
        (data['close'] < data['open']) & 
        (data['close'].shift(1) > data['open'].shift(1)) &
        (data['close'] < data['open'].shift(1)) & 
        (data['open'] > data['close'].shift(1))
    )
    
    # Pattern scores
    data['hammer_score'] = hammer_condition.astype(int)
    data['shooting_star_score'] = shooting_star_condition.astype(int)
    data['engulfing_score'] = (bullish_engulfing | bearish_engulfing).astype(int)
    
    # Multi-timeframe reversal alignment
    data['pattern_alignment'] = (
        data['hammer_score'] + data['shooting_star_score'] + data['engulfing_score'] +
        data['hammer_score'].shift(1).fillna(0) + 
        data['shooting_star_score'].shift(1).fillna(0) +
        data['engulfing_score'].shift(1).fillna(0)
    )
    
    # Liquidity Flow and Market Impact Analysis
    # Transaction-based liquidity assessment
    data['amount_per_trade'] = data['amount'] / (data['volume'] + 1e-8)
    data['avg_trade_size'] = data['amount_per_trade'].rolling(window=10, min_periods=1).mean()
    data['trade_size_ratio'] = data['amount_per_trade'] / (data['avg_trade_size'] + 1e-8)
    
    # Bid-ask spread proxy using high-low range
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    data['normalized_spread'] = data['spread_proxy'] / data['spread_proxy'].rolling(window=20, min_periods=1).mean()
    
    # Volume persistence during reversal attempts
    data['volume_ma'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_persistence'] = data['volume'] / (data['volume_ma'] + 1e-8)
    
    # Reversal-Liquidity Adaptive Signal Generation
    # Construct reversal probability score
    extreme_price_score = (
        (abs(data['range_expansion'] - 1) * np.sign(data['close_ret'])) +
        (abs(data['momentum_percentile']) * np.sign(-data['close_ret'])) +
        (abs(data['gap_normalized']) * np.sign(-data['open_gap']))
    )
    
    pattern_confirmation_score = data['pattern_alignment'] * extreme_price_score
    
    # Weight by liquidity stress indicators
    liquidity_stress = (
        data['normalized_spread'] + 
        (1 / (data['trade_size_ratio'] + 1e-8)) +  # Inverse relationship
        (1 / (data['volume_persistence'] + 1e-8))
    )
    
    # Final alpha factor construction
    reversal_probability = pattern_confirmation_score * liquidity_stress
    
    # Apply liquidity confirmation framework
    liquidity_filter = (
        (data['trade_size_ratio'] < 0.8) |  # Lower than average trade size
        (data['normalized_spread'] > 1.2) |  # Higher than average spread
        (data['volume_persistence'] < 0.7)    # Lower volume persistence
    )
    
    # Final factor: strong reversal patterns with poor liquidity get highest scores
    alpha_factor = reversal_probability * liquidity_filter.astype(int)
    
    # Clean up and return
    result = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
