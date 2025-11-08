import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume interaction, market efficiency,
    temporal patterns, volume dynamics, and order flow analysis.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Interaction: Divergence Momentum
    # Compare 5-day price momentum vs 5-day volume momentum
    price_momentum = data['close'].pct_change(5)
    volume_momentum = data['volume'].pct_change(5)
    divergence_momentum = np.sign(price_momentum) * np.sign(volume_momentum)
    
    # Price-Volume Interaction: Gap Confirmation
    # Opening gap vs volume intensity
    gap_size = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    volume_intensity = data['volume'] / data['volume'].rolling(20).mean()
    gap_confirmation = gap_size * volume_intensity
    
    # Market Efficiency: Range Efficiency
    # Net movement vs total range (efficiency ratio)
    net_movement = (data['close'] - data['open']).abs()
    total_range = (data['high'] - data['low'])
    range_efficiency = np.where(total_range > 0, net_movement / total_range, 0)
    
    # Market Efficiency: Breakout Validation
    # Price breakout from 20-day range with volume confirmation
    high_20 = data['high'].rolling(20).max()
    low_20 = data['low'].rolling(20).min()
    breakout_signal = np.where(data['close'] > high_20.shift(1), 1, 
                              np.where(data['close'] < low_20.shift(1), -1, 0))
    volume_confirmation = data['volume'] / data['volume'].rolling(20).mean()
    breakout_validation = breakout_signal * volume_confirmation
    
    # Temporal Patterns: Multi-Timeframe Convergence
    # Short (5-day), medium (20-day), long (60-day) trend alignment
    short_trend = data['close'] > data['close'].rolling(5).mean()
    medium_trend = data['close'] > data['close'].rolling(20).mean()
    long_trend = data['close'] > data['close'].rolling(60).mean()
    trend_convergence = (short_trend.astype(int) + medium_trend.astype(int) + long_trend.astype(int)) / 3
    
    # Temporal Patterns: Return Component Divergence
    # Close-to-close vs intraday return comparison
    close_to_close = data['close'].pct_change()
    intraday_return = (data['close'] - data['open']) / data['open']
    return_divergence = np.sign(close_to_close) * np.sign(intraday_return)
    
    # Volume Dynamics: Volume-Clustered Persistence
    # Returns during high volume periods (top 20% volume)
    volume_quantile = data['volume'].rolling(50).apply(lambda x: pd.Series(x).quantile(0.8), raw=False)
    high_volume = data['volume'] > volume_quantile
    high_volume_returns = data['close'].pct_change().where(high_volume, 0)
    volume_persistence = high_volume_returns.rolling(5).mean()
    
    # Volume Dynamics: Price Level Persistence
    # Time spent at high-volume price levels (using VWAP)
    vwap = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
    price_deviation = (data['close'] - vwap) / vwap
    # Quick departure (large deviation) indicates momentum
    price_persistence = -price_deviation.rolling(5).std()  # Negative std = quick movement
    
    # Order Flow Analysis: Amount-Volume Imbalance
    # Large amount/small volume = institutional flow
    avg_trade_size = data['amount'] / data['volume']
    avg_trade_size_ratio = avg_trade_size / avg_trade_size.rolling(20).mean()
    order_flow_imbalance = np.where(avg_trade_size_ratio > 1.2, 1, 
                                   np.where(avg_trade_size_ratio < 0.8, -1, 0))
    
    # Combine all components with equal weights
    factor = (
        0.1 * divergence_momentum +
        0.1 * gap_confirmation +
        0.1 * range_efficiency +
        0.1 * breakout_validation +
        0.1 * trend_convergence +
        0.1 * return_divergence +
        0.1 * volume_persistence +
        0.1 * price_persistence +
        0.1 * order_flow_imbalance
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(100).mean()) / factor.rolling(100).std()
    
    return factor
