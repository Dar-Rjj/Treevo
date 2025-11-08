import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns for different timeframes
    data['return_1d'] = data['close'].pct_change(1)
    data['return_3d'] = data['close'].pct_change(3)
    data['return_5d'] = data['close'].pct_change(5)
    
    # Momentum Acceleration Measurement
    data['short_term_accel'] = (data['return_1d'].shift(1) - data['return_3d'].shift(1)) / (data['return_3d'].shift(1) + 1e-8)
    data['medium_term_accel'] = (data['return_3d'] - data['return_5d']) / (data['return_5d'] + 1e-8)
    data['accel_consistency'] = np.sign(data['short_term_accel']) * np.sign(data['medium_term_accel'])
    
    # Volatility-Adaptive Scaling
    data['high_10d'] = data['high'].rolling(window=10, min_periods=1).max()
    data['low_10d'] = data['low'].rolling(window=10, min_periods=1).min()
    data['price_range_10d'] = (data['high_10d'] - data['low_10d']) / data['close']
    
    # Average True Range calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_20d'] = data['tr'].rolling(window=20, min_periods=1).mean()
    data['volatility_regime'] = data['price_range_10d'] / (data['atr_20d'] / data['close'] + 1e-8)
    
    # Volume Trend Confirmation
    def linear_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i >= window - 1:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = (window * np.sum(x*y) - np.sum(x) * np.sum(y)) / (window * np.sum(x**2) - np.sum(x)**2)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['volume_slope_5d'] = linear_slope(data['volume'], 5)
    data['volume_slope_10d'] = linear_slope(data['volume'], 10)
    data['volume_momentum_alignment'] = np.sign(data['volume_slope_5d']) * np.sign(data['short_term_accel'])
    
    # Volatility-Weighted Reversal
    data['vol_weighted_reversal_1d'] = data['return_1d'].shift(1) / (data['price_range_10d'] + 1e-8)
    data['vol_weighted_reversal_3d'] = data['return_3d'] / (data['price_range_10d'] + 1e-8)
    data['vol_weighted_reversal_5d'] = data['return_5d'] / (data['price_range_10d'] + 1e-8)
    
    # Acceleration persistence
    data['accel_persistence'] = data['short_term_accel'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[-1])) / len(x) if len(x) > 0 else 0
    )
    
    # Acceleration-Enhanced Signals
    acceleration_magnitude = (abs(data['short_term_accel']) + abs(data['medium_term_accel'])) / 2
    data['enhanced_reversal_1d'] = data['vol_weighted_reversal_1d'] * acceleration_magnitude * data['accel_consistency']
    data['enhanced_reversal_3d'] = data['vol_weighted_reversal_3d'] * acceleration_magnitude * data['accel_consistency']
    data['enhanced_reversal_5d'] = data['vol_weighted_reversal_5d'] * acceleration_magnitude * data['accel_persistence']
    
    # Volume-Confirmed Components
    volume_strength = (abs(data['volume_slope_5d']) + abs(data['volume_slope_10d'])) / 2
    data['volume_confirmed_1d'] = data['enhanced_reversal_1d'] * data['volume_slope_5d'] * data['volume_momentum_alignment']
    data['volume_confirmed_3d'] = data['enhanced_reversal_3d'] * data['volume_slope_10d'] * data['volume_momentum_alignment']
    data['volume_confirmed_5d'] = data['enhanced_reversal_5d'] * volume_strength * data['volume_momentum_alignment']
    
    # Filter opposing signals
    def filter_opposing_signals(rev_signal, vol_signal):
        return np.where(
            np.sign(rev_signal) == np.sign(vol_signal),
            rev_signal * vol_signal,
            rev_signal * 0.1  # Reduce conflicting signals
        )
    
    data['filtered_1d'] = filter_opposing_signals(data['enhanced_reversal_1d'], data['volume_slope_5d'])
    data['filtered_3d'] = filter_opposing_signals(data['enhanced_reversal_3d'], data['volume_slope_10d'])
    data['filtered_5d'] = filter_opposing_signals(data['enhanced_reversal_5d'], volume_strength)
    
    # Volatility regime adjustment
    volatility_weight = 1 / (1 + abs(data['volatility_regime'] - 1))
    
    # Final Alpha Construction
    weights = np.array([0.4, 0.35, 0.25])  # Short, medium, long-term weights
    components = np.column_stack([
        data['filtered_1d'],
        data['filtered_3d'], 
        data['filtered_5d']
    ])
    
    # Apply volatility regime filtering
    filtered_components = components * volatility_weight.values.reshape(-1, 1)
    
    # Weighted combination
    alpha = np.sum(filtered_components * weights, axis=1)
    
    # Final signal strength adjustment
    signal_strength = (acceleration_magnitude + volume_strength + volatility_weight) / 3
    final_alpha = alpha * signal_strength
    
    return pd.Series(final_alpha, index=data.index)
