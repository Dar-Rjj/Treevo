import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor using multiple technical heuristics
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # 1. Multi-Timeframe Momentum Divergence
    # Short-term (3-day) Price Momentum
    short_momentum = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    # Medium-term (5-day) Price Momentum
    medium_momentum = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Long-term (10-day) Price Momentum
    long_momentum = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Short-term (3-day) Volume Momentum
    volume_momentum = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    
    # Momentum Divergence Composite
    momentum_divergence = (short_momentum - medium_momentum) * (medium_momentum - long_momentum) * volume_momentum
    
    # 2. Volatility-Adjusted Multi-Period Efficiency
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 3-day True Range Average
    tr_avg = true_range.rolling(window=3).mean()
    
    # 5-day Net Price Movement
    net_price_movement = data['close'] - data['close'].shift(5)
    
    # 3-day Cumulative Volume
    cumulative_volume = data['volume'].rolling(window=3).sum()
    
    # Multi-Dimensional Efficiency
    volatility_efficiency = (net_price_movement / tr_avg) * cumulative_volume
    
    # 3. Volume-Weighted Breakout/Support System
    # 5-day Resistance Level
    resistance_level = data['high'].rolling(window=5).max()
    
    # 5-day Support Level
    support_level = data['low'].rolling(window=5).min()
    
    # 3-day Volume Acceleration
    volume_acceleration = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    
    # Volume-Confirmed Breakout/Support Signal
    breakout_signal = ((data['close'] - resistance_level) - (data['close'] - support_level)) * volume_acceleration
    
    # 4. Gap Analysis with Multi-Period Volatility Context
    # Current Gap Magnitude
    gap_magnitude = data['open'] - data['close'].shift(1)
    
    # 3-day Average Daily Range
    daily_range = data['high'] - data['low']
    avg_daily_range = daily_range.rolling(window=3).mean()
    
    # 5-day Gap Persistence
    gap_threshold = 0.5 * (data['high'].shift(1) - data['low'].shift(1))
    gap_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            gap_count = 0
            for j in range(5):
                if abs(data['open'].iloc[i-j] - data['close'].iloc[i-j-1]) > gap_threshold.iloc[i-j-1]:
                    gap_count += 1
            gap_persistence.iloc[i] = gap_count
    
    # Volatility-Contextualized Gap Signal
    gap_signal = (gap_magnitude / avg_daily_range) * gap_persistence
    
    # 5. Amount-Price Integration Across Timeframes
    # 3-day Cumulative Amount
    cumulative_amount = data['amount'].rolling(window=3).sum()
    
    # 5-day Price Range Efficiency
    max_high_5d = data['high'].rolling(window=5).max()
    min_low_5d = data['low'].rolling(window=5).min()
    price_range_efficiency = (data['close'] - data['close'].shift(5)) / (max_high_5d - min_low_5d)
    
    # 3-day Volume Trend
    volume_trend = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    
    # Multi-Timeframe Amount Efficiency
    amount_efficiency = cumulative_amount * price_range_efficiency * volume_trend
    
    # 6. Momentum-Volatility Convergence Divergence
    # 5-day Price Momentum
    price_momentum_5d = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # 5-day Volatility Momentum
    vol_current = data['close'].rolling(window=5).std()
    vol_previous = data['close'].shift(5).rolling(window=5).std()
    volatility_momentum = (vol_current - vol_previous) / vol_previous
    
    # 3-day Volume Momentum (reuse from earlier)
    volume_momentum_3d = volume_momentum
    
    # Convergence-Divergence Signal
    convergence_signal = price_momentum_5d * volatility_momentum * volume_momentum_3d
    
    # 7. Multi-Period Range Breakout with Volume Confirmation
    # 10-day Price Range
    max_high_10d = data['high'].rolling(window=10).max()
    min_low_10d = data['low'].rolling(window=10).min()
    price_range_10d = max_high_10d - min_low_10d
    
    # Current Position in Range
    position_in_range = (data['close'] - min_low_10d) / price_range_10d
    
    # 5-day Volume Surge Indicator
    volume_avg_4d = data['volume'].shift(1).rolling(window=4).mean()
    volume_surge = data['volume'] / volume_avg_4d
    
    # Range-Breakout Probability
    range_breakout = position_in_range * volume_surge
    
    # Combine all factors with equal weighting
    factors = [
        momentum_divergence,
        volatility_efficiency,
        breakout_signal,
        gap_signal,
        amount_efficiency,
        convergence_signal,
        range_breakout
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=data.index)
    for f in factors:
        f_normalized = (f - f.mean()) / f.std()
        combined_factor += f_normalized
    
    # Final normalization
    factor = (combined_factor - combined_factor.mean()) / combined_factor.std()
    
    return factor
