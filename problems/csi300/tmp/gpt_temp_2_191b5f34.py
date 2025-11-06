import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # High-Low Momentum Divergence
    high_momentum = (df['high'] - df['high'].shift(5)) / df['high'].shift(5)
    low_momentum = (df['low'] - df['low'].shift(5)) / df['low'].shift(5)
    divergence = high_momentum - low_momentum
    volume_ratio = df['volume'] / df['volume'].shift(5)
    factor1 = divergence * volume_ratio
    
    # Volatility-Adjusted Price Reversal
    short_term_return = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    overnight_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    high_low_range = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    factor2 = short_term_return * true_range - overnight_return
    
    # Volume-Weighted Price Acceleration
    return_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    return_2d = (df['close'] - df['close'].shift(2)) / df['close'].shift(2)
    acceleration = return_1d - return_2d
    
    volume_rank = df['volume'].rolling(window=20).rank()
    weighted_acceleration = acceleration * volume_rank
    
    price_deviation = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    factor3 = weighted_acceleration * price_deviation
    
    # Amplitude-Volume Correlation Factor
    daily_range = (df['high'] - df['low']) / df['close']
    volume_change = df['volume'] / df['volume'].shift(1)
    
    def rolling_corr(x, y):
        return x.rolling(window=5).corr(y.shift(1))
    
    correlation = rolling_corr(daily_range, volume_change)
    momentum_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    factor4 = correlation * momentum_3d
    
    # Pressure-Release Indicator
    pressure = (df['close'] - df['low']) / (df['high'] - df['low'])
    normalized_pressure = pressure * df['volume']
    avg_pressure_3d = normalized_pressure.rolling(window=3).mean()
    factor5 = normalized_pressure - avg_pressure_3d
    
    # Gap-Fill Probability Factor
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    fill_ratio = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    def calc_historical_fill(gap_series, fill_series, window=10):
        fill_prob = pd.Series(index=gap_series.index, dtype=float)
        for i in range(window, len(gap_series)):
            window_gaps = gap_series.iloc[i-window:i]
            window_fills = fill_series.iloc[i-window:i]
            fill_prob.iloc[i] = (window_fills > 0.5).mean()
        return fill_prob
    
    historical_fill_prob = calc_historical_fill(overnight_gap, fill_ratio)
    gap_persistence = overnight_gap.rolling(window=10).std()
    factor6 = overnight_gap * historical_fill_prob / gap_persistence
    
    # Volume-Implied Momentum Breakout
    volume_mean = df['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    volume_std = df['volume'].rolling(window=20, min_periods=1).std().shift(1)
    volume_zscore = (df['volume'] - volume_mean) / volume_std
    volume_surge = (volume_zscore > 2).astype(float)
    
    high_20d = df['high'].rolling(window=20).max()
    breakout_confirmation = (df['high'] == high_20d).astype(float)
    close_position = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    factor7 = volume_surge * breakout_confirmation * close_position
    
    # Combine all factors with equal weighting
    factors = pd.concat([factor1, factor2, factor3, factor4, factor5, factor6, factor7], axis=1)
    factors.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7']
    
    # Normalize each factor by z-score
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    
    # Equal-weighted combination
    final_factor = normalized_factors.mean(axis=1)
    
    return final_factor
