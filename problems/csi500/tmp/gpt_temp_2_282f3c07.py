import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Volatility-Adjusted Gap Momentum
    gap_magnitude = np.abs(data['open'] / data['close'].shift(1) - 1)
    intraday_vol = (data['high'] - data['low']) / data['close']
    vol_adj_gap = gap_magnitude / intraday_vol * np.sign(data['open'] / data['close'].shift(1) - 1)
    
    # Volume-Weighted Price Acceleration
    price_accel = (data['close'] / data['close'].shift(5) - 1) - (data['close'].shift(5) / data['close'].shift(10) - 1)
    volume_surge = data['volume'] / data['volume'].rolling(window=10, min_periods=1).mean().shift(1)
    vol_weighted_accel = price_accel * np.power(volume_surge, 1/3)
    
    # Range Efficiency Oscillator
    max_potential_range = data['high'].rolling(window=6, min_periods=1).max() - data['low'].rolling(window=6, min_periods=1).min()
    actual_range_util = (data['high'] - data['low']).rolling(window=6, min_periods=1).mean()
    range_efficiency = (1 - actual_range_util / max_potential_range) * (data['close'] / data['close'].shift(1) - 1)
    
    # Pressure Diffusion Indicator
    pressure_gradient = (data['close'] - data['low']) / (data['high'] - data['low'])
    pressure_momentum = pressure_gradient.rolling(window=6, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    pressure_diffusion = np.tanh(pressure_gradient * pressure_momentum)
    
    # Volume-Clustered Reversal
    low_volume_filter = data['volume'] < data['volume'].rolling(window=16, min_periods=1).quantile(0.2)
    reversal_strength = -(data['close'] / data['close'].shift(1) - 1)
    
    def percentile_rank_func(x):
        if len(x) == 0:
            return 0
        current = x[-1]
        return np.mean(x[:-1] < current) if len(x) > 1 else 0
    
    volume_percentile = data['volume'].rolling(window=16, min_periods=1).apply(percentile_rank_func, raw=True)
    volume_clustered_rev = reversal_strength * volume_percentile
    
    # Amplitude-Modulated Trend
    price_amplitude = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    
    def trend_persistence_func(window):
        if len(window) < 6:
            return 0
        returns = [window[i] / window[i-1] - 1 for i in range(1, len(window))]
        if len(returns) < 5:
            return 0
        x = np.arange(len(returns))
        return np.corrcoef(x, returns)[0, 1] if not np.isnan(np.corrcoef(x, returns)[0, 1]) else 0
    
    trend_persistence = data['close'].rolling(window=6, min_periods=1).apply(trend_persistence_func, raw=True)
    amplitude_mod_trend = (data['close'] / data['close'].shift(5) - 1) * price_amplitude * trend_persistence / np.sqrt(price_amplitude)
    
    # Gap-Fill Momentum Divergence
    gap_fill_progress = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'])
    volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    gap_fill_momentum = np.abs(gap_fill_progress * np.log(volume_ratio)) * (data['open'] > data['close'].shift(1)).astype(float)
    
    # Efficiency-Regime Momentum
    market_efficiency = np.abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    regime_filter = market_efficiency < 0.5
    efficiency_regime_momentum = (data['close'] / data['close'].shift(5) - 1) / market_efficiency * np.power(1 - market_efficiency, 2) * regime_filter.astype(float)
    
    # Pressure-Volume Congestion
    price_compression = (data['high'] - data['low']).rolling(window=6, min_periods=1).max() / (data['high'] - data['low']).rolling(window=6, min_periods=1).min()
    volume_compression = data['volume'].rolling(window=6, min_periods=1).max() / data['volume'].rolling(window=6, min_periods=1).min()
    pressure_volume_congestion = (data['close'] / data['close'].shift(1) - 1) / (price_compression * volume_compression)
    
    # Range-Bound Momentum Capture
    range_position = (data['close'] - data['low'].rolling(window=21, min_periods=1).min()) / (data['high'].rolling(window=21, min_periods=1).max() - data['low'].rolling(window=21, min_periods=1).min())
    range_momentum = range_position - range_position.shift(5)
    range_filter = ((range_position < 0.2) | (range_position > 0.8)).astype(float)
    range_bound_momentum = range_position * range_momentum * range_filter * np.power(range_position, 3)
    
    # Combine all factors with equal weights
    factors = [
        vol_adj_gap, vol_weighted_accel, range_efficiency, pressure_diffusion,
        volume_clustered_rev, amplitude_mod_trend, gap_fill_momentum,
        efficiency_regime_momentum, pressure_volume_congestion, range_bound_momentum
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=data.index)
    for f in factors:
        normalized_f = (f - f.rolling(window=252, min_periods=1).mean()) / f.rolling(window=252, min_periods=1).std()
        combined_factor += normalized_f
    
    return combined_factor / len(factors)
