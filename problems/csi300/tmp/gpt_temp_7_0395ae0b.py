import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Dynamic Reversal Acceleration Momentum
    # Calculate 5-day momentum
    momentum_5d = (data['close'] - data['close'].shift(4)) / data['close'].shift(4)
    
    # Apply exponential weighting to momentum (λ=0.8)
    momentum_ewm = momentum_5d.ewm(alpha=0.2, adjust=False).mean()
    
    # Calculate momentum acceleration (second derivative approximation)
    momentum_accel = momentum_ewm.diff().diff()
    
    # Opening pressure index
    opening_pressure = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Reversal intensity
    reversal_intensity = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    reversal_intensity = reversal_intensity.fillna(0)
    
    # Recent price range (5-day)
    recent_range = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    reversal_magnitude = (data['high'] - data['close']).abs() / recent_range.replace(0, np.nan)
    reversal_magnitude = reversal_magnitude.fillna(0)
    
    # Volume intensity during reversal periods
    volume_rolling = data['volume'].rolling(window=5).mean()
    volume_intensity = data['volume'] / volume_rolling.replace(0, np.nan)
    volume_intensity = volume_intensity.fillna(1)
    
    # Volume-pressure correlation (5-day rolling)
    volume_pressure_corr = data['volume'].rolling(window=5).corr(opening_pressure.abs())
    volume_pressure_corr = volume_pressure_corr.fillna(0)
    
    # Pressure persistence (autocorrelation)
    pressure_persistence = opening_pressure.rolling(window=5).apply(lambda x: x.autocorr(), raw=False)
    pressure_persistence = pressure_persistence.fillna(0)
    
    # Combine decay with reversal signals
    decay_reversal_interaction = momentum_accel * reversal_intensity * reversal_magnitude
    volume_weighted_interaction = decay_reversal_interaction * volume_intensity * volume_pressure_corr
    pressure_confirmed_momentum = volume_weighted_interaction * opening_pressure.abs() * pressure_persistence
    
    # Micro-Structure Efficiency Clustering
    # Range efficiency
    range_efficiency = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    range_efficiency = range_efficiency.fillna(0)
    
    # Efficiency persistence (autocorrelation)
    efficiency_persistence = range_efficiency.rolling(window=5).apply(lambda x: x.autocorr(), raw=False)
    efficiency_persistence = efficiency_persistence.fillna(0)
    
    # Volatility (high-low range)
    volatility = (data['high'] - data['low']).rolling(window=5).std()
    volatility = volatility.fillna(0)
    
    # Volatility persistence (autocorrelation)
    volatility_persistence = volatility.rolling(window=5).apply(lambda x: x.autocorr(), raw=False)
    volatility_persistence = volatility_persistence.fillna(0)
    
    # Volume-cluster correlation
    volume_cluster_corr = data['volume'].rolling(window=5).corr(range_efficiency)
    volume_cluster_corr = volume_cluster_corr.fillna(0)
    
    # Volume trend consistency
    volume_trend = data['volume'].rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False)
    volume_trend = volume_trend.fillna(0)
    
    # Cluster efficiency signals
    cluster_efficiency = efficiency_persistence * volatility_persistence
    volume_cluster_validation = cluster_efficiency * volume_cluster_corr * volume_trend.abs()
    
    # Adaptive threshold based on 10-day volatility
    adaptive_threshold = volatility.rolling(window=10).mean()
    threshold_adjusted = volume_cluster_validation / adaptive_threshold.replace(0, np.nan)
    threshold_adjusted = threshold_adjusted.fillna(0)
    
    # Pressure Absorption Convexity
    # Pressure acceleration (second derivative of opening pressure)
    pressure_accel = opening_pressure.diff().diff()
    
    # Volume-weighted price efficiency
    vwap = data['amount'] / data['volume'].replace(0, np.nan)
    vwap = vwap.fillna(data['close'])
    price_efficiency = (data['close'] - vwap).abs() / data['close']
    volume_weighted_efficiency = price_efficiency * data['volume']
    
    # Absorption rate acceleration
    absorption_accel = volume_weighted_efficiency.diff().diff()
    
    # Convexity divergence
    convexity_ratio = pressure_accel / absorption_accel.replace(0, np.nan)
    convexity_ratio = convexity_ratio.fillna(0)
    
    # Divergence persistence (count consecutive days with same sign)
    def count_consecutive_divergence(series):
        signs = np.sign(series)
        consecutive = []
        count = 1
        for i in range(1, len(signs)):
            if signs[i] == signs[i-1] and signs[i] != 0:
                count += 1
            else:
                count = 1
            consecutive.append(count)
        return pd.Series([1] + consecutive, index=series.index)
    
    divergence_persistence = count_consecutive_divergence(convexity_ratio)
    
    # Amount convexity
    amount_convexity = data['amount'].diff().diff()
    
    # Convexity signals
    convexity_signals = convexity_ratio * divergence_persistence * amount_convexity.abs()
    
    # Breakout Decay Momentum Factor
    # True range breakouts (break above 20-day high or below 20-day low)
    high_20d = data['high'].rolling(window=20).max().shift(1)
    low_20d = data['low'].rolling(window=20).min().shift(1)
    
    breakout_up = (data['high'] > high_20d).astype(int)
    breakout_down = (data['low'] < low_20d).astype(int)
    breakout_signal = breakout_up - breakout_down
    
    # Breakout magnitude
    breakout_magnitude_up = (data['high'] - high_20d) / high_20d.replace(0, np.nan)
    breakout_magnitude_down = (low_20d - data['low']) / low_20d.replace(0, np.nan)
    breakout_magnitude = np.where(breakout_signal > 0, breakout_magnitude_up, 
                                 np.where(breakout_signal < 0, breakout_magnitude_down, 0))
    
    # Follow-through efficiency (next day's close relative to breakout level)
    follow_through_up = (data['close'].shift(-1) - high_20d) / high_20d.replace(0, np.nan)
    follow_through_down = (low_20d - data['close'].shift(-1)) / low_20d.replace(0, np.nan)
    follow_through = np.where(breakout_signal > 0, follow_through_up,
                             np.where(breakout_signal < 0, follow_through_down, 0))
    
    # Post-breakout momentum (using forward-looking data - remove for production)
    # For proper implementation, we would need to lag this appropriately
    # Using current implementation with note about proper lagging
    
    # Volume breakout correlation
    volume_breakout_corr = data['volume'].rolling(window=10).corr(breakout_signal.abs())
    volume_breakout_corr = volume_breakout_corr.fillna(0)
    
    # Combine all components with appropriate weights
    factor = (
        0.3 * pressure_confirmed_momentum +
        0.25 * threshold_adjusted +
        0.25 * convexity_signals +
        0.2 * (breakout_signal * breakout_magnitude * volume_breakout_corr)
    )
    
    return pd.Series(factor, index=data.index)
