import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Momentum Framework
    # Price Acceleration: (close_t/close_{t-3} - close_t/close_{t-10}) / 3
    price_acceleration = ((data['close'] / data['close'].shift(3)) - 
                         (data['close'] / data['close'].shift(10))) / 3
    
    # Volume Momentum: volume_t / volume_{t-5} - 1
    volume_momentum = data['volume'] / data['volume'].shift(5) - 1
    
    # Range Momentum: (close_t - low_t) / (high_t - low_t)
    range_momentum = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Cumulative Pressure: sum((close_t - open_t) / (high_t - low_t), 3)
    pressure_component = (data['close'] - data['open']) / (data['high'] - data['low'])
    cumulative_pressure = pressure_component.rolling(window=3, min_periods=1).sum()
    
    # Volatility Breakout Detection
    # Daily Range: high_t - low_t
    daily_range = data['high'] - data['low']
    
    # Historical Volatility: avg(high - low, 10)
    historical_volatility = daily_range.rolling(window=10, min_periods=1).mean()
    
    # Breakout Ratio: daily_range / historical_volatility
    breakout_ratio = daily_range / historical_volatility
    
    # Breakout Signal: breakout_ratio > 1.5
    breakout_signal = (breakout_ratio > 1.5).astype(float)
    
    # Volume-Pressure Regime Classification
    # Volume Regime: volume_t / avg(volume, 10) > 1.2
    volume_avg_10 = data['volume'].rolling(window=10, min_periods=1).mean()
    volume_regime = (data['volume'] / volume_avg_10 > 1.2)
    
    # Pressure Regime: cumulative_pressure > avg(cumulative_pressure, 10)
    pressure_avg_10 = cumulative_pressure.rolling(window=10, min_periods=1).mean()
    pressure_regime = cumulative_pressure > pressure_avg_10
    
    # Regime Types
    high_regime = volume_regime & pressure_regime
    low_regime = (~volume_regime) & (~pressure_regime)
    mixed_regime = ~(high_regime | low_regime)
    
    # Regime-Adaptive Enhancement
    # High Regime Enhancement
    volume_avg_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_confirmation = data['volume'] / volume_avg_5
    
    pressure_std_10 = cumulative_pressure.rolling(window=10, min_periods=1).std()
    pressure_strength = cumulative_pressure / pressure_std_10.replace(0, 1)
    
    range_breakout = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Low Regime Enhancement
    trend_persistence = np.sign(data['close'] - data['close'].shift(1)).rolling(window=5, min_periods=1).sum()
    
    liquidity_quality = data['amount'] / data['volume'].replace(0, 1)
    
    failed_breakout_condition = (data['high'] > data['high'].shift(1)) & (data['close'] < data['open'])
    failed_breakouts = failed_breakout_condition.rolling(window=3, min_periods=1).sum()
    
    # Asymmetric Volume Analysis
    # Up Volume Response: volume_t when (close_t > close_{t-1})
    up_volume_response = np.where(data['close'] > data['close'].shift(1), data['volume'], 0)
    
    # Down Volume Response: volume_t when (close_t < close_{t-1})
    down_volume_response = np.where(data['close'] < data['close'].shift(1), data['volume'], 0)
    
    # Volume Asymmetry: up_volume_response / down_volume_response
    volume_asymmetry = up_volume_response / down_volume_response.replace(0, 1)
    
    # Volume Cluster Detection: count(volume_t > avg(volume, 10) * 1.5, 5)
    volume_cluster_condition = data['volume'] > (volume_avg_10 * 1.5)
    volume_cluster_detection = volume_cluster_condition.rolling(window=5, min_periods=1).sum()
    
    # Composite Alpha Generation
    # Base Momentum: price_acceleration × volume_momentum × range_momentum
    base_momentum = price_acceleration * volume_momentum * range_momentum
    
    # Pressure Multiplier: cumulative_pressure × breakout_signal
    pressure_multiplier = cumulative_pressure * breakout_signal
    
    # Volume Asymmetry Factor: volume_asymmetry × volume_cluster_detection
    volume_asymmetry_factor = volume_asymmetry * volume_cluster_detection
    
    # Regime Application
    high_regime_factor = (base_momentum * volume_confirmation * 
                         pressure_strength * range_breakout)
    
    low_regime_factor = (base_momentum * trend_persistence * 
                        liquidity_quality * (1 - failed_breakouts / 3))
    
    mixed_regime_factor = base_momentum * pressure_multiplier * volume_asymmetry_factor
    
    # Final Alpha: regime_enhanced_factor
    final_alpha = np.where(high_regime, high_regime_factor,
                          np.where(low_regime, low_regime_factor, mixed_regime_factor))
    
    return pd.Series(final_alpha, index=data.index)
