import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Divergence
    # Short-term momentum (t-3 to t)
    short_price_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    intraday_strength = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Medium-term momentum (t-10 to t)
    medium_price_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Trend consistency: Average intraday strength over t-4 to t
    trend_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window_data = []
            for j in range(i-4, i+1):
                if j >= 0:
                    high_val = df['high'].iloc[j]
                    low_val = df['low'].iloc[j]
                    close_val = df['close'].iloc[j]
                    if high_val != low_val:
                        window_data.append((close_val - low_val) / (high_val - low_val))
            if window_data:
                trend_consistency.iloc[i] = np.mean(window_data)
    
    # Divergence Analysis
    momentum_divergence = short_price_momentum - medium_price_momentum
    direction_alignment = np.sign(short_price_momentum) * np.sign(medium_price_momentum)
    
    # Filter condition: Apply only when short-term > medium-term momentum
    momentum_filter = short_price_momentum > medium_price_momentum
    
    # Volume-Price Alignment Detection
    # Volume Momentum Analysis
    short_volume_change = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    medium_volume_change = (df['volume'] - df['volume'].shift(10)) / df['volume'].shift(10)
    
    # Volume concentration: Current volume / average of last 5 volumes
    volume_concentration = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 5:
            avg_volume = df['volume'].iloc[i-5:i].mean()
            if avg_volume > 0:
                volume_concentration.iloc[i] = df['volume'].iloc[i] / avg_volume
    
    # Price-Volume Divergence
    volume_price_divergence = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            if short_price_momentum.iloc[i] > 0 and short_volume_change.iloc[i] < 0:
                volume_price_divergence.iloc[i] = 1.0  # Positive divergence
            elif short_price_momentum.iloc[i] < 0 and short_volume_change.iloc[i] > 0:
                volume_price_divergence.iloc[i] = -1.0  # Negative divergence
            else:
                volume_price_divergence.iloc[i] = 0.0  # Zero divergence
    
    # Intraday Range and Volatility Assessment
    # Current Range Analysis
    normalized_range = (df['high'] - df['low']) / df['close']
    range_momentum = (normalized_range / normalized_range.shift(1)) - 1
    
    # Volatility Profile
    true_range = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 1:
            high_low = df['high'].iloc[i] - df['low'].iloc[i]
            high_close_prev = abs(df['high'].iloc[i] - df['close'].iloc[i-1])
            low_close_prev = abs(df['low'].iloc[i] - df['close'].iloc[i-1])
            true_range.iloc[i] = max(high_low, high_close_prev, low_close_prev) / df['close'].iloc[i]
    
    # Rolling volatility: 20-day average of true range
    rolling_volatility = true_range.rolling(window=20, min_periods=1).mean()
    
    # Mean Reversion Integration
    # Price Deviation Component
    rolling_mean = df['close'].rolling(window=20, min_periods=1).mean().shift(1)
    price_deviation = (df['close'] - rolling_mean) / df['close']
    
    # Volatility Scaling
    rolling_std = df['close'].rolling(window=20, min_periods=1).std().shift(1)
    normalized_deviation = price_deviation / rolling_std
    
    # Adaptive Factor Synthesis
    # Momentum-Divergence Core
    raw_divergence = momentum_divergence * volume_price_divergence
    volume_confirmation = raw_divergence * volume_concentration
    
    # Regime-Adaptive Weighting
    regime_weighted = volume_confirmation * range_momentum / (rolling_volatility + 1e-8)
    bounded_output = np.tanh(regime_weighted)
    
    # Mean Reversion Enhancement
    final_factor = bounded_output * normalized_deviation * direction_alignment
    
    # Apply momentum filter
    final_factor = final_factor.where(momentum_filter, 0)
    
    return final_factor
