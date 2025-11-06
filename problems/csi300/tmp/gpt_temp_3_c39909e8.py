import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining regime-adaptive momentum, asymmetric liquidity flow, 
    and multi-timeframe volatility analysis.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Regime-Adaptive Momentum
    # Price trend detection
    ma_5 = data['close'].rolling(window=5, min_periods=5).mean()
    ma_20 = data['close'].rolling(window=20, min_periods=20).mean()
    price_trend = (ma_5 > ma_20).astype(int)
    
    # Volume regime detection
    volume_10d_max = data['volume'].rolling(window=10, min_periods=10).max()
    volume_10d_min = data['volume'].rolling(window=10, min_periods=10).min()
    volume_position = (data['volume'] - volume_10d_min) / (volume_10d_max - volume_10d_min + 1e-8)
    volume_regime = (volume_position > 0.7).astype(int)  # High volume if in top 30% of 10-day range
    
    # Regime classification
    trending_high_volume = (price_trend == 1) & (volume_regime == 1)
    ranging_low_volume = (price_trend == 0) & (volume_regime == 0)
    transition = ~trending_high_volume & ~ranging_low_volume
    
    # Regime-specific signals
    # Trending High-Volume: (Close_t - Close_{t-2}) / Range_{t-2}
    close_t_minus_2 = data['close'].shift(2)
    range_t_minus_2 = data['high'].shift(2) - data['low'].shift(2) + 1e-8
    trending_signal = (data['close'] - close_t_minus_2) / range_t_minus_2
    
    # Ranging Low-Volume: (Close_t - 5-day Low) / (5-day High - 5-day Low)
    low_5d = data['low'].rolling(window=5, min_periods=5).min()
    high_5d = data['high'].rolling(window=5, min_periods=5).max()
    range_5d = high_5d - low_5d + 1e-8
    ranging_signal = (data['close'] - low_5d) / range_5d
    
    # Transition: Current Range / 10-day Average Range
    current_range = data['high'] - data['low']
    avg_range_10d = current_range.rolling(window=10, min_periods=10).mean()
    transition_signal = current_range / (avg_range_10d + 1e-8)
    
    # Combine regime signals
    regime_momentum = pd.Series(0.0, index=data.index)
    regime_momentum[trending_high_volume] = trending_signal[trending_high_volume]
    regime_momentum[ranging_low_volume] = ranging_signal[ranging_low_volume]
    regime_momentum[transition] = transition_signal[transition]
    
    # 2. Asymmetric Liquidity Flow
    # Pressure measurement
    up_move = data['close'] > data['open']
    down_move = data['close'] < data['open']
    
    buying_volume = np.where(up_move, data['volume'], 0)
    selling_volume = np.where(down_move, data['volume'], 0)
    
    buying_pressure = buying_volume / (data['volume'] + 1e-8)
    selling_pressure = selling_volume / (data['volume'] + 1e-8)
    
    # Net pressure
    net_pressure = (buying_pressure - selling_pressure)
    
    # Flow persistence
    pressure_direction = np.sign(net_pressure)
    persistence = pd.Series(0, index=data.index)
    for i in range(1, len(data)):
        if pressure_direction.iloc[i] == pressure_direction.iloc[i-1] and pressure_direction.iloc[i] != 0:
            persistence.iloc[i] = persistence.iloc[i-1] + 1
    
    liquidity_flow = net_pressure * (1 + persistence / 10.0)
    
    # 3. Multi-Timeframe Volatility
    # Volatility clustering
    # Short-term: Intraday Range / Open
    short_term_vol = (data['high'] - data['low']) / (data['open'] + 1e-8)
    
    # Medium-term: 5-day Range Stability (inverse of coefficient of variation)
    range_5d_rolling = (data['high'] - data['low']).rolling(window=5, min_periods=5)
    range_mean = range_5d_rolling.mean()
    range_std = range_5d_rolling.std()
    medium_term_stability = range_mean / (range_std + 1e-8)
    
    # Long-term: 20-day Volatility Percentile
    volatility_20d = data['close'].pct_change().rolling(window=20, min_periods=20).std()
    vol_percentile = volatility_20d.rolling(window=50, min_periods=50).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Cross-timeframe interaction
    # Compression/Expansion: Short/Medium Volatility Ratio
    vol_ratio = short_term_vol / (medium_term_stability + 1e-8)
    
    # Regime coherence: Volatility pattern consistency
    vol_trend_5d = short_term_vol.rolling(window=5, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    vol_trend_20d = short_term_vol.rolling(window=20, min_periods=20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    regime_coherence = np.sign(vol_trend_5d) == np.sign(vol_trend_20d)
    
    volatility_factor = (vol_ratio * vol_percentile * regime_coherence.astype(float))
    
    # Combine all components with equal weights
    alpha_factor = (
        0.4 * regime_momentum +
        0.35 * liquidity_flow +
        0.25 * volatility_factor
    )
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=20, min_periods=20).mean()) / \
                   (alpha_factor.rolling(window=20, min_periods=20).std() + 1e-8)
    
    return alpha_factor
