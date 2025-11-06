import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    # Price Momentum
    close_momentum_3d = df['close'] / df['close'].shift(3) - 1
    close_momentum_10d = df['close'] / df['close'].shift(10) - 1
    
    # Volume Momentum
    volume_momentum_3d = df['volume'] / df['volume'].shift(3) - 1
    volume_momentum_10d = df['volume'] / df['volume'].shift(10) - 1
    
    # Divergence Signal
    price_dir_3d = np.sign(close_momentum_3d)
    price_dir_10d = np.sign(close_momentum_10d)
    volume_dir_3d = np.sign(volume_momentum_3d)
    volume_dir_10d = np.sign(volume_momentum_10d)
    
    divergence_signal = (price_dir_3d == volume_dir_3d).astype(int) + (price_dir_10d == volume_dir_10d).astype(int)
    
    # High-Low Range Efficiency
    # True Range
    true_range = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                   abs(df['low'] - df['close'].shift(1))))
    
    # Price Movement
    price_movement_5d = abs(df['close'] / df['close'].shift(5) - 1)
    
    # Efficiency Ratio
    efficiency_ratio = price_movement_5d / (true_range.rolling(window=5).sum() / df['close'].shift(5))
    
    # Volume-Weighted Acceleration
    # Price Acceleration
    returns_1d = df['close'].pct_change()
    price_acceleration = returns_1d - returns_1d.shift(3)
    
    # Volume Rank
    volume_rank = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    # Weighted Signal
    weighted_signal = price_acceleration * volume_rank
    
    # Opening Gap Persistence
    # Gap Size
    gap_size = (df['open'] / df['close'].shift(1) - 1)
    
    # Gap Filling
    gap_filled_up = (df['low'] <= df['close'].shift(1)).astype(int)  # Gap up but filled down
    gap_filled_down = (df['high'] >= df['close'].shift(1)).astype(int)  # Gap down but filled up
    
    # Persistence Signal
    gap_persistence = gap_size * (1 - (gap_filled_up + gap_filled_down) / 2)
    
    # Volatility-Adjusted Momentum
    # Parkinson Volatility
    parkinson_vol = np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
    parkinson_vol_5d = np.sqrt(parkinson_vol.rolling(window=5).sum())
    
    # Momentum Magnitude
    momentum_10d = df['close'] / df['close'].shift(10) - 1
    
    # Regime Signal
    regime_signal = momentum_10d / (parkinson_vol_5d + 1e-8)
    
    # Volume-Breakout Quality
    # Key Levels
    high_20d = df['high'].rolling(window=20).max()
    low_20d = df['low'].rolling(window=20).min()
    
    # Breakout Volume
    volume_avg_20d = df['volume'].rolling(window=20).mean()
    breakout_volume_ratio = df['volume'] / volume_avg_20d
    
    # Breakout Signal
    upper_breakout = (df['close'] > high_20d.shift(1)).astype(int) * breakout_volume_ratio
    lower_breakout = (df['close'] < low_20d.shift(1)).astype(int) * breakout_volume_ratio
    breakout_signal = upper_breakout - lower_breakout
    
    # Intraday Strength Pattern
    # Range Position
    range_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Multi-day Consistency
    strength_trend = range_position.rolling(window=5).apply(lambda x: pd.Series(x).rank().iloc[-1] - pd.Series(x).rank().iloc[0], raw=False)
    
    # Persistence Signal
    persistence_signal = range_position * strength_trend
    
    # Volume-Context Mean Reversion
    # Price Deviation
    ma_10d = df['close'].rolling(window=10).mean()
    price_deviation = (df['close'] - ma_10d) / ma_10d
    
    # Volume Comparison
    volume_avg_10d = df['volume'].rolling(window=10).mean()
    volume_ratio = df['volume'] / volume_avg_10d
    
    # Reversion Signal
    reversion_signal = -price_deviation * volume_ratio
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'divergence': divergence_signal,
        'efficiency': efficiency_ratio,
        'weighted_accel': weighted_signal,
        'gap_persistence': gap_persistence,
        'regime': regime_signal,
        'breakout': breakout_signal,
        'intraday_strength': persistence_signal,
        'mean_reversion': reversion_signal
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / (x.rolling(window=20).std() + 1e-8))
    
    # Equal-weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
