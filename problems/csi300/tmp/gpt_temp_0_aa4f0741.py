import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a novel alpha factor combining multiple technical heuristics
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Divergence
    short_momentum = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_momentum = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    volume_divergence = (short_momentum - medium_momentum) * df['volume']
    momentum_signal = np.sign(short_momentum) * np.sign(medium_momentum) * volume_divergence
    
    # Volatility-Normalized Range Efficiency
    price_range = df['high'] - df['low']
    volatility_5d = df['close'].rolling(window=5).std()
    vol_adjusted_efficiency = price_range / volatility_5d.replace(0, np.nan)
    efficiency_score = vol_adjusted_efficiency * df['volume']
    
    # Hierarchical Breakout Detection
    high_5d = df['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan, raw=True)
    low_5d = df['low'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].min() if len(x) > 1 else np.nan, raw=True)
    upper_break = (df['close'] > high_5d).astype(float)
    lower_break = (df['close'] < low_5d).astype(float)
    
    volume_avg_5d = df['volume'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].mean() if len(x) > 1 else np.nan, raw=True)
    volume_spike = (df['volume'] > 1.5 * volume_avg_5d).astype(float)
    breakout_score = (upper_break - lower_break) * volume_spike
    
    # Gap Analysis with Range Context
    gap_magnitude = abs(df['open'] - df['close'].shift(1))
    prev_day_range = df['high'].shift(1) - df['low'].shift(1)
    gap_to_range_ratio = gap_magnitude / prev_day_range.replace(0, np.nan)
    gap_signal = gap_to_range_ratio * df['volume']
    
    # Multi-scale Price-Volume Trend
    def calculate_pvt(window):
        returns = df['close'].pct_change().fillna(0)
        pvt_values = []
        for i in range(len(df)):
            if i >= window-1:
                window_returns = returns.iloc[i-window+1:i+1]
                window_volume = df['volume'].iloc[i-window+1:i+1]
                pvt = (window_returns * window_volume).sum()
            else:
                pvt = np.nan
            pvt_values.append(pvt)
        return pd.Series(pvt_values, index=df.index)
    
    short_pvt = calculate_pvt(3)
    medium_pvt = calculate_pvt(10)
    trend_divergence = short_pvt - medium_pvt
    
    # Amount-Based Market Impact
    price_movement = df['close'].diff()
    movement_per_amount = price_movement / df['amount'].replace(0, np.nan)
    avg_impact_5d = movement_per_amount.rolling(window=5).mean()
    impact_deviation = movement_per_amount - avg_impact_5d
    
    # Support/Resistance Efficiency
    high_5d_current = df['high'].rolling(window=5).max()
    low_5d_current = df['low'].rolling(window=5).min()
    dist_to_high = (df['close'] - high_5d_current) / high_5d_current.replace(0, np.nan)
    dist_to_low = (df['close'] - low_5d_current) / low_5d_current.replace(0, np.nan)
    range_efficiency_ratio = abs(dist_to_high - dist_to_low) / (df['high'] - df['low']).replace(0, np.nan)
    efficiency_signal = range_efficiency_ratio * df['volume']
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'momentum': momentum_signal,
        'efficiency': efficiency_score,
        'breakout': breakout_score,
        'gap': gap_signal,
        'trend': trend_divergence,
        'impact': impact_deviation,
        'support_resistance': efficiency_signal
    })
    
    # Normalize each factor and combine
    normalized_factors = factors.apply(lambda x: (x - x.mean()) / x.std())
    combined_factor = normalized_factors.mean(axis=1)
    
    return combined_factor
