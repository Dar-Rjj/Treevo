import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Regime Momentum-Breakout Alignment with Volume-Price Dynamics
    """
    data = df.copy()
    
    # Dual-Timeframe Momentum Analysis
    # Price acceleration
    mom_5d = data['close'] / data['close'].shift(5) - 1
    mom_10d = data['close'] / data['close'].shift(10) - 1
    price_acceleration = mom_5d - mom_10d
    
    # Volume acceleration
    vol_mom_5d = data['volume'] / data['volume'].shift(5) - 1
    vol_mom_10d = data['volume'] / data['volume'].shift(10) - 1
    volume_acceleration = vol_mom_5d - vol_mom_10d
    
    # Momentum divergence patterns
    positive_divergence = ((price_acceleration > 0) & (volume_acceleration < 0)).astype(int)
    negative_divergence = ((price_acceleration < 0) & (volume_acceleration > 0)).astype(int)
    convergence = ((price_acceleration * volume_acceleration) > 0).astype(int)
    
    # Breakout Asymmetry and Range Dynamics
    # Breakout asymmetry
    upward_breakouts = (data['close'] > data['high'].shift(1)).rolling(window=10, min_periods=5).sum()
    downward_breakouts = (data['close'] < data['low'].shift(1)).rolling(window=10, min_periods=5).sum()
    net_breakout_asymmetry = upward_breakouts - downward_breakouts
    
    # Range efficiency
    intraday_range_utilization = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    high_5d = data['high'].rolling(window=5, min_periods=3).max()
    low_5d = data['low'].rolling(window=5, min_periods=3).min()
    range_expansion = (high_5d - low_5d) / data['close'].shift(5).replace(0, np.nan)
    
    range_5d = data['high'].rolling(window=5, min_periods=3).max() - data['low'].rolling(window=5, min_periods=3).min()
    range_10d = data['high'].rolling(window=10, min_periods=5).max() - data['low'].rolling(window=10, min_periods=5).min()
    range_momentum = range_5d / range_10d.replace(0, np.nan) - 1
    
    # Breakout-Range composite
    breakout_range_base = net_breakout_asymmetry * intraday_range_utilization
    breakout_range_scaled = breakout_range_base * range_expansion
    breakout_range_filtered = breakout_range_scaled * np.sign(range_momentum)
    
    # Intraday Pressure and Gap Dynamics
    # Intraday pressure
    bullish_pressure = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    bearish_pressure = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    net_pressure = bullish_pressure - bearish_pressure
    
    # Gap analysis
    daily_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1).replace(0, np.nan)
    
    # Gap direction consistency (3-day)
    gap_direction = np.sign(daily_gap)
    gap_consistency = gap_direction.rolling(window=3, min_periods=2).apply(
        lambda x: 1 if len(set(x.dropna())) == 1 else 0, raw=False
    )
    
    # Gap follow-through
    gap_follow_through = abs(data['close'] - data['open']) / abs(data['open'] - data['close'].shift(1)).replace(0, np.nan)
    
    # Volume ratio
    volume_ratio = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Intraday composite
    intraday_base = net_pressure * volume_ratio
    intraday_enhanced = intraday_base * gap_consistency * abs(daily_gap)
    intraday_composite = intraday_enhanced * gap_follow_through
    
    # Volatility Regime Detection
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    vol_5d = true_range.rolling(window=5, min_periods=3).std()
    vol_20d = true_range.rolling(window=20, min_periods=10).std()
    vol_ratio = vol_5d / vol_20d.replace(0, np.nan)
    
    # Regime classification
    high_vol_regime = (vol_ratio > 1.2).astype(int)
    low_vol_regime = (vol_ratio < 0.8).astype(int)
    transitional_regime = ((vol_ratio >= 0.8) & (vol_ratio <= 1.2)).astype(int)
    
    # Component Scores
    # Momentum divergence score
    momentum_score = positive_divergence - negative_divergence
    momentum_score = momentum_score * abs(price_acceleration - volume_acceleration)
    
    # Breakout-range score
    breakout_score = breakout_range_filtered
    
    # Intraday composite score
    intraday_score = intraday_composite
    
    # Regime-weighted combination
    high_vol_signal = (
        0.5 * momentum_score + 
        0.3 * breakout_score + 
        0.2 * intraday_score
    )
    
    low_vol_signal = (
        0.2 * momentum_score + 
        0.3 * breakout_score + 
        0.5 * intraday_score
    )
    
    transitional_signal = (
        (momentum_score + breakout_score + intraday_score) / 3
    )
    
    # Final adaptive signal
    final_signal = (
        high_vol_regime * high_vol_signal +
        low_vol_regime * low_vol_signal +
        transitional_regime * transitional_signal
    )
    
    # Volume confirmation
    volume_persistence = (volume_acceleration.rolling(window=3, min_periods=2).mean() > 0).astype(int)
    volume_alignment = ((np.sign(price_acceleration) == np.sign(volume_acceleration)).astype(int) * 2 - 1)
    
    # Enhanced signal with volume confirmation
    enhanced_signal = final_signal * (1 + 0.2 * volume_alignment * volume_persistence)
    
    # Normalize and return
    alpha_factor = enhanced_signal / enhanced_signal.rolling(window=20, min_periods=10).std().replace(0, np.nan)
    
    return alpha_factor
