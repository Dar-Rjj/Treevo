import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple technical indicators:
    - Relative Strength of High-Low Range to Volume
    - Price Momentum with Volume Confirmation
    - Intraday Volatility Persistence
    - Volume-Weighted Price Acceleration
    - Amount-Based Order Flow Imbalance
    - Close-to-Open Gap Fill Probability
    - High-Low Compression Breakout
    - Volume-Weighted Price Reversal
    """
    
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Relative Strength of High-Low Range to Volume
    high_low_range = data['high'] - data['low']
    volume_adjusted_range = high_low_range / (data['volume'] + 1e-8)
    range_percentile = volume_adjusted_range.rolling(window=20).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    short_term_avg = volume_adjusted_range.rolling(window=5).mean()
    long_term_avg = volume_adjusted_range.rolling(window=20).mean()
    range_strength = (short_term_avg - long_term_avg) * range_percentile
    
    # 2. Price Momentum with Volume Confirmation
    price_momentum = data['close'].pct_change(periods=10)
    volume_trend = data['volume'].pct_change(periods=10)
    momentum_volume_divergence = np.sign(price_momentum) * np.sign(volume_trend)
    momentum_strength = price_momentum.abs() * volume_trend.abs() * momentum_volume_divergence
    
    # 3. Intraday Volatility Persistence
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    def autocorr_lag1(x):
        if len(x) < 2:
            return np.nan
        return x.autocorr(lag=1)
    
    vol_clustering = true_range.rolling(window=20).apply(autocorr_lag1, raw=False)
    vol_clustering_median = vol_clustering.rolling(window=60).median()
    price_trend = data['close'].pct_change(periods=5)
    volatility_signal = (vol_clustering - vol_clustering_median) * price_trend
    
    # 4. Volume-Weighted Price Acceleration
    price_acceleration = data['close'].pct_change(periods=5).diff(periods=5)
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).std()
    volume_zscore = volume_zscore.replace([np.inf, -np.inf], np.nan).fillna(0)
    acceleration_signal = (price_acceleration * volume_zscore).rolling(window=3).mean()
    
    # 5. Amount-Based Order Flow Imbalance
    price_change = data['close'].diff()
    tick_rule = np.sign(price_change)
    tick_rule = tick_rule.replace(0, np.nan).fillna(method='ffill')
    order_flow = data['amount'] * tick_rule
    
    def consecutive_runs(series):
        if len(series) < 2:
            return 0
        signs = np.sign(series)
        runs = (signs != signs.shift(1)).cumsum()
        current_run = runs.iloc[-1]
        run_length = (runs == current_run).sum()
        return run_length * np.sign(series.iloc[-1])
    
    flow_persistence = order_flow.rolling(window=10).apply(consecutive_runs, raw=False)
    flow_signal = flow_persistence * data['close'].pct_change(periods=5)
    
    # 6. Close-to-Open Gap Fill Probability
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_size = overnight_gap.abs()
    opening_volume_ratio = data['volume'] / data['volume'].rolling(window=20).mean()
    
    def gap_fill_probability(gap_series, volume_ratio):
        if len(gap_series) < 21:
            return 0
        current_gap = gap_series.iloc[-1]
        historical_gaps = gap_series.iloc[:-1]
        if len(historical_gaps) == 0:
            return 0
        fill_threshold = 0.5 * current_gap
        filled_gaps = historical_gaps[historical_gaps.abs() <= abs(fill_threshold)]
        fill_rate = len(filled_gaps) / len(historical_gaps) if len(historical_gaps) > 0 else 0
        volume_adjustment = 1 / (1 + np.exp(-volume_ratio.iloc[-1]))
        return fill_rate * volume_adjustment * (-np.sign(current_gap))
    
    gap_signal = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        window_data = overnight_gap.iloc[:i+1]
        volume_window = opening_volume_ratio.iloc[:i+1]
        gap_signal.iloc[i] = gap_fill_probability(window_data, volume_window)
    
    # 7. High-Low Compression Breakout
    range_compression = high_low_range / high_low_range.rolling(window=20).mean()
    range_midpoint = (data['high'] + data['low']) / 2
    breakout_direction = np.sign(data['close'] - range_midpoint)
    volume_surge = data['volume'] / data['volume'].rolling(window=20).mean()
    breakout_signal = (1 / range_compression) * breakout_direction * volume_surge
    
    # 8. Volume-Weighted Price Reversal
    recent_high = data['high'].rolling(window=10).max()
    recent_low = data['low'].rolling(window=10).min()
    price_extreme = (data['close'] - recent_low) / (recent_high - recent_low + 1e-8)
    volume_extreme = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Overbought/Oversold detection with volume confirmation
    overbought = (price_extreme > 0.8) & (volume_extreme > 1.2)
    oversold = (price_extreme < 0.2) & (volume_extreme > 1.2)
    reversal_signal = pd.Series(0, index=data.index)
    reversal_signal[overbought] = -1
    reversal_signal[oversold] = 1
    reversal_signal = reversal_signal * volume_extreme * (1 - 2 * abs(price_extreme - 0.5))
    
    # Combine all signals with equal weights
    signals = pd.DataFrame({
        'range_strength': range_strength,
        'momentum_strength': momentum_strength,
        'volatility_signal': volatility_signal,
        'acceleration_signal': acceleration_signal,
        'flow_signal': flow_signal,
        'gap_signal': gap_signal,
        'breakout_signal': breakout_signal,
        'reversal_signal': reversal_signal
    })
    
    # Normalize each signal and combine
    normalized_signals = signals.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else x)
    combined_signal = normalized_signals.mean(axis=1)
    
    return combined_signal
