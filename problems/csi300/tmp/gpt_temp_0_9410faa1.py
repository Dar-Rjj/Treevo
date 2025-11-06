import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Dynamic Volatility-Adjusted Price Momentum
    # Calculate Price Momentum
    short_term_return = df['close'] / df['close'].shift(5) - 1
    medium_term_return = df['close'] / df['close'].shift(20) - 1
    price_momentum = (short_term_return + medium_term_return) / 2
    
    # Compute Volatility Adjustment
    high_low_range = (df['high'] - df['low']) / df['close']
    rolling_volatility = high_low_range.rolling(window=20, min_periods=10).std()
    volatility_score = 1 / (1 + rolling_volatility)
    
    # Combine Momentum with Volatility
    volatility_adjusted_momentum = price_momentum * volatility_score
    
    # Volume-Weighted Price Reversal Strength
    # Identify Recent Price Extremes
    local_max = df['high'].rolling(window=5, min_periods=3).max()
    local_min = df['low'].rolling(window=5, min_periods=3).min()
    distance_from_max = (local_max - df['close']) / df['close']
    distance_from_min = (df['close'] - local_min) / df['close']
    
    # Compute Volume Confirmation
    max_volume = df['volume'].rolling(window=5, min_periods=3).max()
    avg_volume = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_ratio = max_volume / avg_volume
    
    # Generate Reversal Signal
    price_extreme_distance = np.where(distance_from_max > distance_from_min, 
                                    distance_from_max, -distance_from_min)
    volume_weighted_reversal = price_extreme_distance * volume_ratio
    
    # Liquidity-Adjusted Breakout Detection
    # Detect Price Breakouts
    resistance_level = df['high'].rolling(window=20, min_periods=10).max()
    support_level = df['low'].rolling(window=20, min_periods=10).min()
    
    resistance_break = (df['close'] - resistance_level.shift(1)) / resistance_level.shift(1)
    support_break = (support_level.shift(1) - df['close']) / support_level.shift(1)
    breakout_magnitude = np.where(resistance_break > 0, resistance_break, 
                                np.where(support_break > 0, -support_break, 0))
    
    # Assess Liquidity Conditions
    volume_surge = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    bid_ask_proxy = (df['high'] - df['low']) / df['close']
    avg_spread = bid_ask_proxy.rolling(window=20, min_periods=10).mean()
    liquidity_score = volume_surge / (1 + bid_ask_proxy / avg_spread)
    
    # Validate Breakout Quality
    liquidity_adjusted_breakout = breakout_magnitude * liquidity_score
    
    # Order Flow Imbalance Factor
    # Calculate Intraday Pressure
    open_close_relation = (df['close'] - df['open']) / df['open']
    high_low_normalized = (df['high'] - df['low']) / df['close']
    pressure_index = open_close_relation / (1 + high_low_normalized)
    
    # Volume Distribution Analysis
    # Using rolling windows to approximate intraday patterns
    morning_volume_ratio = df['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: x.iloc[-1] / x.mean() if len(x) > 0 else 1
    )
    volume_concentration = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    volume_pattern_score = morning_volume_ratio * volume_concentration
    
    # Generate Order Flow Signal
    order_flow_signal = pressure_index * volume_pattern_score
    
    # Regime-Adaptive Mean Reversion
    # Identify Mean Reversion Opportunities
    price_deviation_5 = (df['close'] - df['close'].rolling(window=5, min_periods=3).mean()) / df['close'].rolling(window=5, min_periods=3).std()
    price_deviation_10 = (df['close'] - df['close'].rolling(window=10, min_periods=5).mean()) / df['close'].rolling(window=10, min_periods=5).std()
    price_deviation_20 = (df['close'] - df['close'].rolling(window=20, min_periods=10).mean()) / df['close'].rolling(window=20, min_periods=10).std()
    combined_deviation = (price_deviation_5 + price_deviation_10 + price_deviation_20) / 3
    
    # Assess Market Regime
    volatility_regime = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    trend_strength = df['close'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.std() if x.std() > 0 else 0
    )
    regime_adjustment = 1 / (1 + abs(trend_strength) + volatility_regime)
    
    # Adaptive Mean Reversion Signal
    adaptive_mean_reversion = combined_deviation * regime_adjustment
    
    # Combine all factors with equal weighting
    final_factor = (
        volatility_adjusted_momentum.fillna(0) +
        volume_weighted_reversal.fillna(0) +
        liquidity_adjusted_breakout.fillna(0) +
        order_flow_signal.fillna(0) +
        adaptive_mean_reversion.fillna(0)
    ) / 5
    
    return final_factor
