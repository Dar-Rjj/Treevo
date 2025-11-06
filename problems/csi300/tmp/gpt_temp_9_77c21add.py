import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Volatility Adjusted Momentum
    # Calculate 5-day and 10-day momentum
    momentum_5 = df['close'] / df['close'].shift(5) - 1
    momentum_10 = df['close'] / df['close'].shift(10) - 1
    
    # Calculate daily high-low range and 20-day average
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    avg_range_20 = daily_range.rolling(window=20, min_periods=10).mean()
    
    # Normalize momentum by volatility
    vol_adj_momentum_5 = momentum_5 / avg_range_20
    vol_adj_momentum_10 = momentum_10 / avg_range_20
    
    # Volume-Driven Price Reversal
    # Calculate intraday return and identify volume spikes
    intraday_return = (df['high'] - df['low']) / df['close'].shift(1)
    volume_spike = df['volume'] > df['volume'].rolling(window=20, min_periods=10).quantile(0.9)
    
    # Calculate next 3-day return (using shift to avoid lookahead)
    next_3d_return = df['close'].shift(-3) / df['close'] - 1
    # Remove future-looking data by shifting forward
    reversal_signal = next_3d_return.shift(3) * intraday_return.shift(1) * volume_spike.shift(1)
    
    # Liquidity-Weighted Trend Strength
    # Calculate multiple timeframe trends
    trend_5 = df['close'] / df['close'].rolling(window=5, min_periods=3).mean() - 1
    trend_20 = df['close'] / df['close'].rolling(window=20, min_periods=10).mean() - 1
    trend_60 = df['close'] / df['close'].rolling(window=60, min_periods=30).mean() - 1
    
    # Calculate liquidity proxy (inverted effective spread)
    effective_spread = (df['high'] - df['low']) / df['close']
    liquidity_weight = 1 / (effective_spread.rolling(window=20, min_periods=10).mean() + 1e-6)
    
    # Weight trends by liquidity
    liquidity_trend = (trend_5 + trend_20 + trend_60) / 3 * liquidity_weight
    
    # Opening Gap Persistence Factor
    # Calculate opening gap
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Track gap direction consistency over 10 days
    gap_direction = np.sign(opening_gap)
    consecutive_same_sign = gap_direction.rolling(window=10, min_periods=5).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Volume confirmation
    volume_ratio = df['volume'] / df['volume'].rolling(window=20, min_periods=10).mean()
    gap_persistence = opening_gap * consecutive_same_sign * volume_ratio
    
    # Amount-Based Smart Money Indicator
    # Calculate average trade size and detect clusters
    avg_trade_size = df['amount'] / (df['volume'] + 1e-6)
    large_trade_concentration = avg_trade_size / avg_trade_size.rolling(window=20, min_periods=10).mean()
    
    # Identify unusual concentration in 5-day window
    trade_cluster = large_trade_concentration.rolling(window=5, min_periods=3).apply(
        lambda x: 1 if x.iloc[-1] > x.quantile(0.8) else 0, raw=False
    )
    
    # Correlate with contemporaneous returns
    contemporaneous_return = df['close'] / df['close'].shift(1) - 1
    smart_money_signal = trade_cluster * contemporaneous_return * large_trade_concentration
    
    # Combine all factors with equal weights
    factor = (
        vol_adj_momentum_5.fillna(0) + 
        vol_adj_momentum_10.fillna(0) + 
        reversal_signal.fillna(0) + 
        liquidity_trend.fillna(0) + 
        gap_persistence.fillna(0) + 
        smart_money_signal.fillna(0)
    ) / 6
    
    return factor
