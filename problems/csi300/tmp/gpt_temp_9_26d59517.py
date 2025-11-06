import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Dynamic Liquidity Absorption Factor
    # Calculate Bidirectional Price Impact
    upside_absorption = (df['high'] - df['open']) * df['volume']
    downside_absorption = (df['open'] - df['low']) * df['volume']
    
    # Assess Absorption Efficiency
    net_absorption = np.abs(upside_absorption - downside_absorption)
    total_absorption_capacity = upside_absorption + downside_absorption
    
    # Handle zero division
    absorption_ratio = np.where(total_absorption_capacity > 0, 
                               net_absorption / total_absorption_capacity, 0)
    
    # Preserve sign based on net absorption direction
    absorption_ratio = absorption_ratio * np.sign(upside_absorption - downside_absorption)
    
    # Incorporate Momentum Context
    def calc_slope(window):
        if len(window) < 2:
            return 0
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window)
        return slope
    
    # Calculate recent price trend (5-day slope)
    close_prices = df['close']
    trend_slope = close_prices.rolling(window=5, min_periods=2).apply(calc_slope, raw=False)
    
    liquidity_signal = absorption_ratio * trend_slope
    
    # Volatility Clustering Breakout Detector
    # Identify Volatility Regimes
    short_vol = (df['high'] - df['low']).rolling(window=5, min_periods=1).mean()
    medium_vol = (df['high'] - df['low']).rolling(window=20, min_periods=1).mean()
    
    # Detect Volatility Breakout
    volatility_ratio = short_vol / medium_vol
    
    # Assess Price Confirmation
    bar_strength = np.abs(df['close'] - df['open']) / (df['high'] - df['low'])
    bar_strength = bar_strength.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Combine with volume surge
    volume_avg_10 = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_ratio = df['volume'] / volume_avg_10
    
    # Generate Breakout Signal
    breakout_signal = volatility_ratio * bar_strength * volume_ratio
    
    # Apply directional bias
    directional_bias = np.where(df['close'] > df['open'], 1, -1)
    breakout_signal = breakout_signal * directional_bias
    
    # Momentum Divergence Oscillator
    # Calculate Price Momentum
    fast_momentum = df['close'] / df['close'].shift(2) - 1
    slow_momentum = df['close'] / df['close'].shift(9) - 1
    
    # Assess Momentum Divergence
    momentum_spread = np.abs(fast_momentum - slow_momentum)
    
    # Evaluate Momentum Consistency
    direction_indicator = np.where(fast_momentum * slow_momentum > 0, 1, -1)
    
    # Combine with volume pattern
    def calc_volume_slope(window):
        if len(window) < 2:
            return 0
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window)
        return slope
    
    volume_trend = df['volume'].rolling(window=5, min_periods=2).apply(calc_volume_slope, raw=False)
    
    # Generate Divergence Signal
    divergence_signal = momentum_spread * direction_indicator * volume_trend
    
    # Apply non-linear transformation
    divergence_signal = np.tanh(divergence_signal * 10)  # Enhanced sensitivity
    
    # Price-Volume Congestion Breakout
    # Measure Trading Range Congestion
    atr_10 = (df['high'] - df['low']).rolling(window=10, min_periods=1).mean()
    atr_20 = (df['high'] - df['low']).rolling(window=20, min_periods=1).mean()
    price_compression = atr_10 / atr_20
    
    # Assess Volume Drying
    volume_max_10 = df['volume'].rolling(window=10, min_periods=1).max()
    volume_decline_ratio = df['volume'] / volume_max_10
    
    # Detect Breakout Initiation
    # Price Breakout Confirmation
    current_range = df['high'] - df['low']
    avg_range_10 = current_range.rolling(window=10, min_periods=1).mean()
    breakout_magnitude = current_range / avg_range_10
    
    # Volume Expansion Validation
    volume_avg_congestion = df['volume'].rolling(window=10, min_periods=1).mean()
    volume_expansion = df['volume'] / volume_avg_congestion
    
    # Generate Congestion Signal
    congestion_signal = breakout_magnitude * volume_expansion
    
    # Apply directional weighting
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    range_position = range_position.replace([np.inf, -np.inf], 0.5).fillna(0.5)
    directional_weight = (range_position - 0.5) * 2  # Rescale to -1 to 1
    
    congestion_signal = congestion_signal * directional_weight
    
    # Combine all factors with equal weighting
    final_factor = (
        liquidity_signal.fillna(0) + 
        breakout_signal.fillna(0) + 
        divergence_signal.fillna(0) + 
        congestion_signal.fillna(0)
    ) / 4
    
    return final_factor
