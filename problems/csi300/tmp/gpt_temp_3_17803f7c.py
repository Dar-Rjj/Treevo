import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Momentum-Adjusted Volume Divergence
    # Volume-Weighted Price Momentum
    n = 5  # momentum period
    m = 10  # volume averaging period
    k = 5   # volume trend period
    
    # Calculate price return
    price_return = data['close'] / data['close'].shift(n) - 1
    
    # Calculate volume ratio
    avg_volume = data['volume'].rolling(window=m, min_periods=1).mean().shift(1)
    volume_ratio = data['volume'] / avg_volume
    
    # Volume-weighted price change
    vol_weighted_price_change = price_return * volume_ratio
    
    # High-Low normalized range
    high_low_range = (data['high'] - data['low']) / data['close']
    
    # Adjust momentum by range (avoid division by zero)
    range_adjusted_momentum = vol_weighted_price_change / (high_low_range + 1e-8)
    
    # Volume trend
    volume_trend = data['volume'] / data['volume'].rolling(window=k, min_periods=1).mean().shift(1)
    
    # Volume-Price Divergence
    price_momentum = data['close'] / data['close'].shift(n) - 1
    volume_divergence = volume_trend - price_momentum
    
    # Volatility-Regime Adaptive Factor
    # Realized volatility using high-low range
    realized_vol = (data['high'] - data['low']).rolling(window=20, min_periods=1).std()
    
    # Classify volatility regime
    vol_percentile = realized_vol.rolling(window=60, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)), raw=False
    )
    
    # High volatility regime - mean reversion
    ma_short = data['close'].rolling(window=5, min_periods=1).mean()
    price_deviation = (data['close'] - ma_short) / ma_short
    volume_change = data['volume'] / data['volume'].shift(1)
    mean_reversion_signal = -price_deviation * volume_change
    
    # Liquidity filter
    amount_volume_ratio = data['amount'] / (data['volume'] + 1e-8)
    liquidity_filtered_signal = mean_reversion_signal * amount_volume_ratio
    
    # Low volatility regime - momentum
    price_velocity = data['close'] / data['close'].shift(2) - 1
    velocity_change = price_velocity - price_velocity.shift(2)
    accelerated_momentum = price_velocity * velocity_change
    
    # Volume persistence filter
    volume_consistency = data['volume'].rolling(window=5, min_periods=1).std() / data['volume'].rolling(window=5, min_periods=1).mean()
    momentum_filtered = accelerated_momentum / (volume_consistency + 1e-8)
    
    # Range expansion adjustment
    range_change = high_low_range / high_low_range.rolling(window=10, min_periods=1).mean()
    momentum_adjusted = momentum_filtered * range_change
    
    # Combine regimes
    regime_signal = vol_percentile * liquidity_filtered_signal + (1 - vol_percentile) * momentum_adjusted
    
    # Intraday Pressure Accumulation
    # Opening gap pressure
    gap_pressure = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday strength
    close_to_high = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    volume_intensity = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean()
    intraday_strength = close_to_high * volume_intensity
    
    # Session pressure
    session_pressure = gap_pressure + intraday_strength
    
    # Accumulate multi-day pressure
    p = 3  # pressure accumulation period
    pressure_persistence = session_pressure.rolling(window=p, min_periods=1).sum()
    
    # Pressure breakout detection
    pressure_threshold = pressure_persistence.rolling(window=20, min_periods=1).apply(
        lambda x: np.percentile(x, 80), raw=False
    )
    pressure_breakout = (pressure_persistence > pressure_threshold).astype(float)
    
    # Liquidity Flow Momentum
    # Volume-Amount efficiency
    volume_amount_efficiency = data['amount'] / (data['volume'] + 1e-8)
    
    # Liquidity trend
    q = 5  # liquidity trend period
    liquidity_trend = volume_amount_efficiency / volume_amount_efficiency.rolling(window=q, min_periods=1).mean().shift(1)
    
    # Price response to liquidity
    price_change = data['close'] / data['close'].shift(1) - 1
    price_response = price_change / (liquidity_trend + 1e-8)
    
    # Liquidity-Price divergence
    expected_response = price_response.rolling(window=10, min_periods=1).mean()
    liquidity_divergence = price_response - expected_response
    
    # Range Expansion Momentum
    r = 10  # range comparison period
    current_range = data['high'] - data['low']
    avg_range = current_range.rolling(window=r, min_periods=1).mean().shift(1)
    range_ratio = current_range / (avg_range + 1e-8)
    
    # Expansion threshold
    expansion_threshold = 1.5  # 50% above average
    volatility_breakout = (range_ratio > expansion_threshold).astype(float)
    
    # Directional bias
    close_to_open = (data['close'] - data['open']) / data['open']
    close_to_high_pos = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    close_to_low_pos = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    directional_score = close_to_open * close_to_high_pos * (1 - close_to_low_pos)
    
    # Combine expansion with direction
    range_momentum = volatility_breakout * directional_score
    
    # Final factor combination with weights
    factor = (
        0.25 * range_adjusted_momentum +
        0.20 * volume_divergence +
        0.15 * regime_signal +
        0.15 * pressure_breakout +
        0.15 * liquidity_divergence +
        0.10 * range_momentum
    )
    
    return factor
