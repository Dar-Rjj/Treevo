import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Price Range Efficiency with Volume Confirmation
    # Calculate Intraday Efficiency
    intraday_efficiency = (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 0.001)
    
    # Assess Volume Confirmation
    volume_median_5d = data['volume'].rolling(window=5, min_periods=3).median()
    volume_ratio = data['volume'] / volume_median_5d
    
    # Generate Efficiency Factor
    efficiency_factor = intraday_efficiency * volume_ratio
    
    # Volatility-Regime Adjusted Momentum
    # Identify Volatility State
    price_range_5d = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    price_range_20d = (data['high'] - data['low']).rolling(window=20, min_periods=10).mean()
    volatility_ratio = price_range_5d / price_range_20d
    
    # Compute Regime-Adjusted Momentum
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    regime_adjusted_momentum = momentum_3d / volatility_ratio
    
    # Liquidity Breakout Detection
    # Measure Liquidity Shock
    volume_acceleration = data['volume'] / data['volume'].shift(1) - 1
    volume_acceleration_range = volume_acceleration.rolling(window=10, min_periods=5).apply(
        lambda x: x.max() - x.min() if len(x.dropna()) >= 5 else np.nan
    )
    
    # Assess Price Response
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 0.001)
    
    # Generate Breakout Factor
    breakout_factor = volume_acceleration * abs(intraday_strength)
    
    # Opening Gap Persistence
    # Calculate Gap Magnitude
    gap_magnitude = abs((data['open'] - data['close'].shift(1)) / data['close'].shift(1))
    
    # Assess Intraday Fade/Follow
    gap_close_ratio = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 0.001 * data['close'].shift(1))
    gap_close_ratio = gap_close_ratio.clip(-2, 2)  # Limit extreme values
    
    # Generate Gap Factor
    gap_factor = gap_magnitude * (1 - abs(gap_close_ratio)) * data['volume']
    
    # Multi-Timeframe Volume Divergence
    # Short-term Volume Trend (3-day)
    def volume_slope(series):
        if len(series.dropna()) < 3:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    volume_trend = data['volume'].rolling(window=3, min_periods=3).apply(volume_slope, raw=False)
    
    # Medium-term Volume Level (10-day)
    def volume_percentile(series):
        if len(series.dropna()) < 5:
            return np.nan
        current = series.iloc[-1]
        return (current - series.min()) / (series.max() - series.min() + 0.001)
    
    volume_percentile_10d = data['volume'].rolling(window=10, min_periods=5).apply(volume_percentile, raw=False)
    
    # Generate Divergence Signal
    divergence_signal = volume_trend * (1 - volume_percentile_10d)
    
    # Close-to-Close vs Intraday Momentum
    # Calculate Overnight Momentum
    overnight_momentum = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Calculate Intraday Momentum
    intraday_momentum = (data['close'] - data['open']) / data['open']
    
    # Generate Momentum Factor
    momentum_factor = (intraday_momentum - overnight_momentum) * data['volume']
    
    # Price Compression Breakout
    # Measure Price Compression
    daily_range_3d = ((data['high'] - data['low']) / data['close']).rolling(window=3, min_periods=2).mean()
    daily_range_10d = ((data['high'] - data['low']) / data['close']).rolling(window=10, min_periods=5).mean()
    compression_ratio = daily_range_3d / daily_range_10d
    
    # Detect Breakout Trigger
    current_range_ratio = (data['high'] - data['low']) / data['close']
    expansion_ratio = current_range_ratio / daily_range_3d
    
    # Generate Breakout Factor
    price_breakout_factor = compression_ratio * expansion_ratio * data['volume']
    
    # Volume-Weighted Price Efficiency
    # Calculate Price Noise
    price_noise = (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 0.001)
    
    # Compute Volume Significance
    dollar_volume = data['volume'] * data['close']
    dollar_volume_median_10d = dollar_volume.rolling(window=10, min_periods=5).median()
    volume_significance = dollar_volume / dollar_volume_median_10d
    
    # Generate Efficiency Score
    efficiency_score = price_noise / volume_significance
    
    # Combine all factors with equal weights
    combined_factor = (
        efficiency_factor.fillna(0) +
        regime_adjusted_momentum.fillna(0) +
        breakout_factor.fillna(0) +
        gap_factor.fillna(0) +
        divergence_signal.fillna(0) +
        momentum_factor.fillna(0) +
        price_breakout_factor.fillna(0) +
        efficiency_score.fillna(0)
    )
    
    return combined_factor
