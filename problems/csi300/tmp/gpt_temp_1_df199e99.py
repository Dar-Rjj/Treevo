import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Volatility Adjusted Momentum
    intraday_momentum = (df['high'] - df['low']) / df['close']
    vol_adj_momentum = intraday_momentum / df['close'].rolling(window=5).std()
    
    # Volume-Scaled Price Reversal
    price_reversal = -1 * (df['close'] / df['close'].shift(1) - 1)
    volume_scaling = df['volume'] / df['volume'].rolling(window=20).mean()
    volume_scaled_reversal = price_reversal * volume_scaling
    
    # Amplitude-Weighted Trend Strength
    price_amplitude = (df['high'] - df['low']) / df['open']
    
    def linear_regression_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
        return pd.Series(slopes, index=series.index)
    
    trend_component = linear_regression_slope(df['close'], 10)
    amplitude_trend = price_amplitude * trend_component
    
    # Volume-Price Divergence Factor
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    price_momentum = df['close'] / df['close'].shift(5) - 1
    divergence_factor = volume_momentum - price_momentum
    
    # Efficiency Ratio Adjusted Return
    price_change_10 = df['close'] - df['close'].shift(10)
    abs_price_changes = abs(df['close'] - df['close'].shift(1))
    sum_abs_changes = abs_price_changes.rolling(window=10).sum()
    efficiency_ratio = price_change_10 / sum_abs_changes
    recent_return = (df['close'] / df['close'].shift(3) - 1) * efficiency_ratio
    
    # Pressure-Based Reversal Indicator
    buying_pressure = (df['close'] - df['low']) / (df['high'] - df['low'])
    pressure_volume = buying_pressure * df['volume']
    pressure_reversal = -1 * (df['close'] / df['close'].shift(1) - 1) * pressure_volume
    
    # Range Breakout Confidence
    high_20 = df['high'].rolling(window=20).max()
    breakout_signal = (df['high'] > high_20.shift(1)).astype(float)
    volume_avg_20 = df['volume'].rolling(window=20).mean()
    strength_measure = ((df['high'] - high_20.shift(1)) / high_20.shift(1)) * (df['volume'] / volume_avg_20)
    breakout_confidence = breakout_signal * strength_measure
    
    # Liquidity-Adjusted Momentum
    price_momentum_10 = df['close'] / df['close'].shift(10) - 1
    liquidity_adjustment = df['volume'] / df['amount']
    liquidity_momentum = price_momentum_10 * liquidity_adjustment
    
    # Volatility Regime Adaptive Factor
    def average_true_range(df, window):
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    atr_20 = average_true_range(df, 20)
    atr_median = atr_20.rolling(window=100).median()
    
    high_vol_regime = -1 * (df['close'] - df['close'].rolling(window=5).mean())
    low_vol_regime = df['close'] / df['close'].shift(5) - 1
    
    volatility_factor = np.where(atr_20 > atr_median, high_vol_regime, low_vol_regime)
    volatility_factor = pd.Series(volatility_factor, index=df.index)
    
    # Volume-Weighted Price Levels
    volume_rank = df['volume'].rolling(window=20, min_periods=1).rank(ascending=False)
    high_key_level = np.where(volume_rank <= 3, df['high'], np.nan)
    low_key_level = np.where(volume_rank <= 3, df['low'], np.nan)
    
    def nearest_distance(price, levels):
        distances = []
        for i in range(len(price)):
            valid_levels = levels[:i+1][~np.isnan(levels[:i+1])]
            if len(valid_levels) > 0:
                min_dist = min(abs(price.iloc[i] - level) for level in valid_levels)
                distances.append(min_dist)
            else:
                distances.append(np.nan)
        return pd.Series(distances, index=price.index)
    
    distance_to_high = nearest_distance(df['close'], high_key_level)
    distance_to_low = nearest_distance(df['close'], low_key_level)
    nearest_distance_level = pd.concat([distance_to_high, distance_to_low], axis=1).min(axis=1)
    
    volume_avg = df['volume'].rolling(window=20).mean()
    volume_weighted_levels = nearest_distance_level * (df['volume'] / volume_avg)
    
    # Combine all factors with equal weights
    factors = [
        vol_adj_momentum,
        volume_scaled_reversal,
        amplitude_trend,
        divergence_factor,
        recent_return,
        pressure_reversal,
        breakout_confidence,
        liquidity_momentum,
        volatility_factor,
        volume_weighted_levels
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        normalized_factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
        combined_factor = combined_factor + normalized_factor.fillna(0)
    
    return combined_factor
