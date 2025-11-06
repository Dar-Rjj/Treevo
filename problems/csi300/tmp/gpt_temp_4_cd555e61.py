import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Acceleration with Volume Confirmation
    # Short-Term Momentum (5-day return)
    short_momentum = df['close'] / df['close'].shift(5) - 1
    
    # Medium-Term Momentum (15-day return from t-20 to t-5)
    medium_momentum = df['close'].shift(5) / df['close'].shift(20) - 1
    
    # Momentum Acceleration
    momentum_acceleration = (short_momentum - medium_momentum) / (abs(medium_momentum) + 1e-8)
    
    # Volume Confirmation
    volume_ratio = df['volume'].rolling(5).mean() / (df['volume'].shift(5).rolling(15).mean() + 1e-8)
    factor1 = momentum_acceleration * volume_ratio
    
    # High-Low Range Breakout Efficiency
    # True Range
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - prev_close)
    tr3 = abs(df['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Price Movement (5-day absolute return)
    price_movement = abs(df['close'] / df['close'].shift(5) - 1)
    
    # Range Efficiency
    range_efficiency = price_movement / (true_range.rolling(5).sum() + 1e-8)
    
    # Volatility Adjustment
    vol_5d = df['close'].pct_change().rolling(5).std()
    vol_20d = df['close'].pct_change().rolling(20).std()
    vol_ratio = vol_5d / (vol_20d + 1e-8)
    factor2 = range_efficiency * vol_ratio
    
    # Volume-Price Divergence Strength
    # Price Trend (10-day linear regression slope)
    def linear_slope(series):
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    price_trend = df['close'].rolling(10).apply(linear_slope, raw=False)
    
    # Volume Trend (10-day linear regression slope)
    volume_trend = df['volume'].rolling(10).apply(linear_slope, raw=False)
    
    # Divergence
    divergence = price_trend * volume_trend
    
    # Scale by Price Level
    price_level = df['close'].rolling(10).mean()
    factor3 = divergence / (price_level + 1e-8)
    
    # Opening Gap Persistence Factor
    # Opening Gap
    opening_gap = df['open'] / df['close'].shift(1) - 1
    
    # Gap Filling
    def calculate_gap_filling(row):
        if row['opening_gap'] > 0:  # Gap up
            gap_filled = max(0, (row['high'] - row['open']) / (abs(row['opening_gap']) * df['close'].shift(1).iloc[0] + 1e-8))
        else:  # Gap down
            gap_filled = max(0, (row['open'] - row['low']) / (abs(row['opening_gap']) * df['close'].shift(1).iloc[0] + 1e-8))
        return min(gap_filled, 1.0)
    
    gap_df = df.copy()
    gap_df['opening_gap'] = opening_gap
    gap_filling = gap_df.apply(calculate_gap_filling, axis=1)
    
    # Gap Persistence
    gap_persistence = abs(opening_gap - gap_filling * np.sign(opening_gap))
    
    # Volume Intensity
    volume_intensity = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    factor4 = gap_persistence * volume_intensity
    
    # Volatility-Adjusted Mean Reversion
    # Price Deviation
    ma_20 = df['close'].rolling(20).mean()
    price_deviation = (df['close'] - ma_20) / ma_20
    
    # Normalized Deviation
    std_20 = df['close'].rolling(20).std() / ma_20
    normalized_deviation = price_deviation / (std_20 + 1e-8)
    
    # Reversion Signal
    reversion_signal = -np.tanh(normalized_deviation)
    
    # Volume Acceleration Multiplier
    volume_acceleration = df['volume'] / (df['volume'].shift(5) + 1e-8) - 1
    factor5 = reversion_signal * volume_acceleration
    
    # Combine all factors with equal weights
    combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
    
    return combined_factor
