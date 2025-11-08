import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    # Calculate Directional Price Movement
    close = df['close']
    volume = df['volume']
    
    # Price trend slopes
    price_trend_5 = close.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    price_trend_10 = close.rolling(window=10).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    
    # Volume trend slopes
    volume_trend_5 = volume.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    volume_trend_10 = volume.rolling(window=10).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    
    # Compare price and volume trend directions
    trend_direction_match_5 = np.sign(price_trend_5) * np.sign(volume_trend_5)
    trend_direction_match_10 = np.sign(price_trend_10) * np.sign(volume_trend_10)
    
    # Generate Divergence Signal
    divergence_signal_5 = price_trend_5 * trend_direction_match_5
    divergence_signal_10 = price_trend_10 * trend_direction_match_10
    
    divergence_strength = (divergence_signal_5 + divergence_signal_10) / 2
    pv_divergence = np.arcsinh(divergence_strength)
    
    # Intraday Momentum Persistence
    open_price = df['open']
    high = df['high']
    low = df['low']
    
    # Intraday Price Patterns
    intraday_range = high - low
    intraday_move = close - open_price
    range_efficiency = intraday_move / intraday_range.replace(0, np.nan)
    range_efficiency = range_efficiency.fillna(0)
    
    close_to_open_persistence = (close - open_price.shift(1)) / open_price.shift(1).replace(0, np.nan)
    close_to_open_persistence = close_to_open_persistence.fillna(0)
    
    # Multi-day Pattern Continuation
    pattern_3day = close.rolling(window=3).apply(lambda x: len([i for i in range(1, len(x)) if np.sign(x[i] - x[i-1]) == np.sign(x[2] - x[1])]) / 2, raw=True)
    pattern_5day = close.rolling(window=5).apply(lambda x: len([i for i in range(1, len(x)) if np.sign(x[i] - x[i-1]) == np.sign(x[4] - x[3])]) / 4, raw=True)
    
    pattern_strength = pattern_3day + pattern_5day
    
    # Volume Confirmation
    volume_persistence = volume.rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0, raw=True)
    volume_persistence = volume_persistence.fillna(0)
    
    price_persistence = (range_efficiency + close_to_open_persistence + pattern_strength) / 3
    intraday_momentum = np.cbrt(price_persistence * volume_persistence.replace(0, 1e-10))
    
    # Volatility Breakout Efficiency
    # Volatility Compression Periods
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    
    atr_10 = true_range(high, low, close.shift(1)).rolling(window=10).mean()
    atr_20 = true_range(high, low, close.shift(1)).rolling(window=20).mean()
    
    volatility_compression = atr_10 / atr_20.replace(0, 1e-10)
    
    # Breakout Signals
    breakout_magnitude = (close - close.rolling(window=5).mean()) / close.rolling(window=5).std().replace(0, 1e-10)
    breakout_direction = close.rolling(window=3).apply(lambda x: len([i for i in range(1, len(x)) if np.sign(x[i] - x[i-1]) == np.sign(x[2] - x[1])]) / 2, raw=True)
    
    # Breakout Quality
    volume_ratio = volume / volume.rolling(window=10).mean().replace(0, 1e-10)
    breakout_quality = breakout_magnitude * breakout_direction * volume_ratio
    volatility_breakout = np.arctanh(np.clip(breakout_quality, -0.99, 0.99))
    
    # Price Level Memory Effect
    # Historical Resistance/Support
    def find_key_levels(prices, window=20):
        levels = []
        for i in range(len(prices) - window + 1):
            window_data = prices.iloc[i:i+window]
            # Simple approach: use local maxima/minima
            if len(window_data) >= 3:
                if window_data.iloc[1] > window_data.iloc[0] and window_data.iloc[1] > window_data.iloc[2]:
                    levels.append(window_data.iloc[1])
                elif window_data.iloc[1] < window_data.iloc[0] and window_data.iloc[1] < window_data.iloc[2]:
                    levels.append(window_data.iloc[1])
        return levels[-5:] if levels else [prices.iloc[-1]]
    
    resistance_levels = high.rolling(window=20).apply(lambda x: find_key_levels(pd.Series(x))[-1] if len(find_key_levels(pd.Series(x))) > 0 else x[-1], raw=True)
    support_levels = low.rolling(window=20).apply(lambda x: find_key_levels(pd.Series(x))[-1] if len(find_key_levels(pd.Series(x))) > 0 else x[-1], raw=True)
    
    # Current Price Proximity
    distance_to_resistance = (close - resistance_levels) / close.replace(0, 1e-10)
    distance_to_support = (close - support_levels) / close.replace(0, 1e-10)
    
    approach_momentum = close.pct_change(3)
    proximity_score = np.minimum(abs(distance_to_resistance), abs(distance_to_support)) * np.sign(approach_momentum)
    
    # Generate Reversal Probability
    volume_trend = volume.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    reversal_probability = 1 / (1 + np.exp(-(proximity_score * volume_trend.replace(0, 1e-10))))
    
    # Volume-Flow Price Acceleration
    # Volume-Weighted Price Changes
    vwap = (close * volume).rolling(window=5).sum() / volume.rolling(window=5).sum().replace(0, 1e-10)
    volume_weighted_momentum = (vwap - vwap.shift(3)) / vwap.shift(3).replace(0, 1e-10)
    
    volume_flow = volume.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0, raw=True)
    
    # Acceleration Patterns
    short_term_accel = volume_weighted_momentum.diff(2)
    medium_term_accel = volume_weighted_momentum.diff(5)
    acceleration_ratio = short_term_accel / medium_term_accel.replace(0, 1e-10)
    
    acceleration_persistence = volume_weighted_momentum.rolling(window=3).apply(lambda x: len([i for i in range(1, len(x)) if np.sign(x[i] - x[i-1]) == np.sign(x[2] - x[1])]) / 2, raw=True)
    
    # Momentum Quality Score
    volatility_env = close.rolling(window=10).std() / close.rolling(window=10).mean().replace(0, 1e-10)
    momentum_sustainability = (acceleration_ratio * volume_flow * acceleration_persistence) / volatility_env.replace(0, 1e-10)
    
    volume_flow_acceleration = np.sign(momentum_sustainability) * np.power(abs(momentum_sustainability), 1/3)
    
    # Combine all factors with equal weighting
    final_factor = (pv_divergence + intraday_momentum + volatility_breakout + reversal_probability + volume_flow_acceleration) / 5
    
    return final_factor
