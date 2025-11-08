import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Regime Adjusted Momentum
    # Compute Short-Term Momentum
    momentum = df['close'] / df['close'].shift(5) - 1
    
    # Compute Volatility Regime
    # Calculate True Range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Rolling Volatility (20-day average of True Range)
    rolling_vol = true_range.rolling(window=20).mean()
    
    # Classify Regime
    vol_60th = rolling_vol.rolling(window=60).quantile(0.6)
    vol_40th = rolling_vol.rolling(window=60).quantile(0.4)
    
    high_vol_regime = rolling_vol > vol_60th
    low_vol_regime = rolling_vol < vol_40th
    
    # Adjust Momentum by Regime
    momentum_adjusted = momentum.copy()
    momentum_adjusted[low_vol_regime] = momentum_adjusted[low_vol_regime] * 1.5
    momentum_adjusted[high_vol_regime] = momentum_adjusted[high_vol_regime] * 0.7
    
    # Volume Acceleration with Price Confirmation
    # Calculate Volume Acceleration
    volume_ma = df['volume'].rolling(window=10).mean()
    volume_accel = df['volume'] / df['volume'].shift(1) - 1
    
    # Check Price Direction Confirmation
    # Calculate price trend slope (3-day)
    def linear_slope(x):
        if len(x) < 3 or np.all(x == x[0]):
            return 0
        return np.polyfit(range(len(x)), x, 1)[0]
    
    price_trend = df['close'].rolling(window=3).apply(linear_slope, raw=True)
    
    # Generate Composite Signal
    volume_signal = volume_accel * price_trend
    
    # Liquidity Barrier Breakthrough
    # Identify Liquidity Barriers
    resistance = df['high'].rolling(window=10).max()
    support = df['low'].rolling(window=10).min()
    
    # Detect Breakthrough Events
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_surge = df['volume'] > (volume_ma_20 * 1.5)
    
    breakthrough_up = (df['close'] > resistance.shift(1)) & volume_surge
    breakthrough_down = (df['close'] < support.shift(1)) & volume_surge
    
    # Score Breakthrough Strength
    breakthrough_strength = pd.Series(0, index=df.index)
    
    # Upward breakthrough
    up_strength = ((df['close'] - resistance.shift(1)) / resistance.shift(1)) * 100
    up_strength = up_strength * (df['volume'] / volume_ma_20).clip(upper=3.0)
    breakthrough_strength[breakthrough_up] = up_strength[breakthrough_up]
    
    # Downward breakthrough (negative)
    down_strength = ((df['close'] - support.shift(1)) / support.shift(1)) * 100
    down_strength = down_strength * (df['volume'] / volume_ma_20).clip(upper=3.0)
    breakthrough_strength[breakthrough_down] = down_strength[breakthrough_down]
    
    # Mean Reversion with Volatility Filtering
    # Calculate Price Deviation
    price_ma_20 = df['close'].rolling(window=20).mean()
    price_std_20 = df['close'].rolling(window=20).std()
    price_deviation = (df['close'] - price_ma_20) / price_std_20
    
    # Apply Volatility Filter
    recent_vol = (df['high'] - df['low']).rolling(window=5).mean()
    vol_filter = recent_vol / recent_vol.rolling(window=20).mean()
    
    # Generate Final Factor (inverted for mean reversion)
    mean_reversion_factor = -price_deviation * vol_filter
    
    # Intraday Strength Persistence
    # Compute Intraday Strength
    daily_range = df['high'] - df['low']
    range_utilization = (df['close'] - df['low']) / daily_range.replace(0, np.nan)
    
    # Measure Strength Consistency
    strong_close = (range_utilization > 0.6).astype(int)
    weak_close = (range_utilization < 0.4).astype(int)
    
    # Count consecutive strong/weak closes
    def count_consecutive(series, condition):
        count = series * 0
        current_streak = 0
        for i in range(len(series)):
            if condition.iloc[i]:
                current_streak += 1
            else:
                current_streak = 0
            count.iloc[i] = current_streak
        return count
    
    strong_streak = count_consecutive(strong_close, strong_close == 1)
    weak_streak = count_consecutive(weak_close, weak_close == 1)
    
    # Combine with Volume Profile (simplified using amount/volume ratio)
    volume_concentration = df['amount'] / df['volume'].replace(0, np.nan)
    volume_weight = volume_concentration / volume_concentration.rolling(window=10).mean()
    
    # Generate Persistence Score
    strength_persistence = (strong_streak - weak_streak) * range_utilization * volume_weight
    
    # Combine all factors with equal weights
    final_factor = (
        momentum_adjusted.fillna(0) * 0.25 +
        volume_signal.fillna(0) * 0.25 +
        breakthrough_strength.fillna(0) * 0.2 +
        mean_reversion_factor.fillna(0) * 0.15 +
        strength_persistence.fillna(0) * 0.15
    )
    
    return final_factor
