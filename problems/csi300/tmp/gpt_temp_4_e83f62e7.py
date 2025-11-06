import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Volume Divergence Momentum
    # Price Momentum (3-day vs 10-day close momentum comparison)
    price_momentum_3d = data['close'].pct_change(3)
    price_momentum_10d = data['close'].pct_change(10)
    price_momentum_signal = price_momentum_3d - price_momentum_10d
    
    # Volume Momentum (3-day vs 10-day volume change comparison)
    volume_momentum_3d = data['volume'].pct_change(3)
    volume_momentum_10d = data['volume'].pct_change(10)
    volume_momentum_signal = volume_momentum_3d - volume_momentum_10d
    
    # Divergence Signal (momentum direction alignment indicator)
    divergence_signal = np.sign(price_momentum_signal) * np.sign(volume_momentum_signal)
    
    # High-Low Range Efficiency
    # True Range (daily high-low-close range)
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Price Movement (5-day absolute return)
    price_movement = abs(data['close'].pct_change(5))
    
    # Efficiency Ratio (movement/range ratio)
    efficiency_ratio = price_movement / (true_range.rolling(window=5).mean() / data['close'])
    
    # Volume-Weighted Acceleration
    # Price Acceleration (return change over 5 days)
    returns_5d = data['close'].pct_change(5)
    price_acceleration = returns_5d.diff(3)
    
    # Volume Rank (20-day percentile position)
    volume_rank = data['volume'].rolling(window=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Weighted Signal (acceleration × volume rank)
    weighted_signal = price_acceleration * volume_rank
    
    # Opening Gap Persistence
    # Gap Size (open vs previous close percentage)
    gap_size = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap Filling (intraday high/low vs gap level)
    gap_filling = np.where(
        gap_size > 0,
        (data['high'] - data['open']) / gap_size.abs(),
        (data['open'] - data['low']) / gap_size.abs()
    )
    
    # Persistence Signal (gap resistance strength)
    persistence_signal = gap_size * (1 - gap_filling)
    
    # Volatility-Adjusted Momentum
    # Parkinson Volatility (5-day high-low based)
    parkinson_vol = np.log(data['high'] / data['low']).rolling(window=5).std()
    
    # Momentum Magnitude (close-based 10-day return)
    momentum_magnitude = data['close'].pct_change(10)
    
    # Regime Signal (momentum/volatility ratio)
    regime_signal = momentum_magnitude / (parkinson_vol + 1e-8)
    
    # Volume-Breakout Quality
    # Key Levels (20-day high/low price boundaries)
    high_20d = data['high'].rolling(window=20).max()
    low_20d = data['low'].rolling(window=20).min()
    
    # Breakout Volume (current vs 20-day average)
    avg_volume_20d = data['volume'].rolling(window=20).mean()
    breakout_volume = data['volume'] / (avg_volume_20d + 1e-8)
    
    # Breakout Signal (price level × volume confirmation)
    price_level = np.where(
        data['close'] > high_20d.shift(1), 1,
        np.where(data['close'] < low_20d.shift(1), -1, 0)
    )
    breakout_signal = price_level * breakout_volume
    
    # Intraday Strength Pattern
    # Range Position (close relative to daily high-low)
    range_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-day Consistency (3-day strength trend)
    strength_trend = range_position.rolling(window=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Persistence Signal (pattern reliability score)
    pattern_signal = range_position * strength_trend
    
    # Volume-Context Mean Reversion
    # Price Deviation (from 10-day moving average)
    ma_10d = data['close'].rolling(window=10).mean()
    price_deviation = (data['close'] - ma_10d) / ma_10d
    
    # Volume Comparison (deviation period vs normal volume)
    normal_volume = data['volume'].rolling(window=20).mean()
    volume_context = data['volume'] / (normal_volume + 1e-8)
    
    # Reversion Signal (deviation × volume context)
    reversion_signal = -price_deviation * volume_context
    
    # Combine all signals with equal weights
    factor = (
        divergence_signal.fillna(0) +
        efficiency_ratio.fillna(0) +
        weighted_signal.fillna(0) +
        persistence_signal.fillna(0) +
        regime_signal.fillna(0) +
        breakout_signal.fillna(0) +
        pattern_signal.fillna(0) +
        reversion_signal.fillna(0)
    )
    
    return factor
