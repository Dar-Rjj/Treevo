import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using volatility-scaled momentum, volume-price divergence,
    intraday multi-session convergence, and multi-timeframe geometric integration.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Volatility-Scaled Momentum
    # Multi-Timeframe Momentum Alignment
    mom_3d = data['close'] / data['close'].shift(3) - 1
    mom_5d = data['close'] / data['close'].shift(5) - 1
    mom_10d = data['close'] / data['close'].shift(10) - 1
    
    # Corresponding Volatility Scaling
    returns = data['close'].pct_change()
    vol_3d = returns.rolling(window=3).std()
    vol_5d = returns.rolling(window=5).std()
    vol_10d = returns.rolling(window=10).std()
    
    # Volatility-scaled momentum and geometric mean combination
    vol_scaled_mom_3d = mom_3d / (vol_3d + 1e-8)
    vol_scaled_mom_5d = mom_5d / (vol_5d + 1e-8)
    vol_scaled_mom_10d = mom_10d / (vol_10d + 1e-8)
    
    momentum_factor = np.sign(vol_scaled_mom_3d * vol_scaled_mom_5d * vol_scaled_mom_10d) * \
                     np.power(np.abs(vol_scaled_mom_3d * vol_scaled_mom_5d * vol_scaled_mom_10d), 1/3)
    
    # Volume-Price Divergence Confirmation
    # Price Strength Geometric Components
    intraday_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    high_low_dominance = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    price_efficiency = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Divergence Multi-Timeframe
    volume_geo_2d = data['volume'].rolling(window=3).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    volume_geo_4d = data['volume'].rolling(window=5).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    
    short_term_volume_ratio = data['volume'] / (volume_geo_2d + 1e-8)
    medium_term_volume_ratio = data['volume'] / (volume_geo_4d + 1e-8)
    volume_acceleration = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # Combined Signal
    core_price_strength = np.sign(intraday_strength * high_low_dominance * price_efficiency) * \
                         np.power(np.abs(intraday_strength * high_low_dominance * price_efficiency), 1/3)
    
    volume_confirmation = np.sign(short_term_volume_ratio * medium_term_volume_ratio * volume_acceleration) * \
                         np.power(np.abs(short_term_volume_ratio * medium_term_volume_ratio * volume_acceleration), 1/3)
    
    volume_price_factor = core_price_strength * volume_confirmation
    
    # Intraday Multi-Session Convergence
    # Morning Session Geometric Factors
    opening_gap_strength = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    morning_range_efficiency = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    morning_pressure = (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Afternoon Session Geometric Factors
    hl_midpoint = (data['high'] + data['low']) / 2
    ohl_avg = (data['open'] + data['high'] + data['low']) / 3
    
    afternoon_strength = (data['close'] - hl_midpoint) / (hl_midpoint + 1e-8)
    closing_momentum = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    session_persistence = (data['close'] - ohl_avg) / (ohl_avg + 1e-8)
    
    # Intraday Alignment
    morning_factors = np.sign(opening_gap_strength * morning_range_efficiency * morning_pressure) * \
                     np.power(np.abs(opening_gap_strength * morning_range_efficiency * morning_pressure), 1/3)
    
    afternoon_factors = np.sign(afternoon_strength * closing_momentum * session_persistence) * \
                       np.power(np.abs(afternoon_strength * closing_momentum * session_persistence), 1/3)
    
    intraday_factor = morning_factors * afternoon_factors * closing_momentum
    
    # Multi-Timeframe Geometric Integration
    # Short-Term Geometric Signals (1-3 days)
    price_momentum_st = data['close'] / data['close'].shift(2) - 1
    
    volume_geo_3d = data['volume'].rolling(window=3).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    volume_price_alignment = (data['volume'] / (volume_geo_3d + 1e-8)) * (data['close'].pct_change())
    
    range_efficiency_st = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)).rolling(window=2).apply(
        lambda x: np.exp(np.mean(np.log(np.abs(x) + 1e-8))))
    
    short_term_signals = np.sign(price_momentum_st * volume_price_alignment * range_efficiency_st) * \
                        np.power(np.abs(price_momentum_st * volume_price_alignment * range_efficiency_st), 1/3)
    
    # Medium-Term Geometric Signals (5-10 days)
    mom_3d_mt = data['close'] / data['close'].shift(3) - 1
    mom_5d_mt = data['close'] / data['close'].shift(5) - 1
    mom_7d_mt = data['close'] / data['close'].shift(7) - 1
    
    price_trend_persistence = np.sign(mom_3d_mt * mom_5d_mt * mom_7d_mt) * \
                             np.power(np.abs(mom_3d_mt * mom_5d_mt * mom_7d_mt), 1/3)
    
    volume_geo_7d = data['volume'].rolling(window=7).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    volume_trend = data['volume'] / (volume_geo_7d + 1e-8)
    
    daily_ranges = (data['high'] - data['low']) / data['close']
    daily_returns = np.abs(data['close'].pct_change())
    
    volatility_efficiency = daily_ranges.rolling(window=7).apply(
        lambda x: np.exp(np.mean(np.log(x + 1e-8)))) / \
        daily_returns.rolling(window=7).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    
    medium_term_signals = np.sign(price_trend_persistence * volume_trend * volatility_efficiency) * \
                         np.power(np.abs(price_trend_persistence * volume_trend * volatility_efficiency), 1/3)
    
    # Final Signal Combination
    multi_timeframe_convergence = np.sign(short_term_signals * medium_term_signals) * \
                                 np.power(np.abs(short_term_signals * medium_term_signals), 1/2)
    
    volume_volatility_alignment = multi_timeframe_convergence * volume_trend / (volatility_efficiency + 1e-8)
    
    # Signal robustness: Geometric mean across all major factor categories
    all_factors = pd.DataFrame({
        'momentum': momentum_factor,
        'volume_price': volume_price_factor,
        'intraday': intraday_factor,
        'multi_timeframe': volume_volatility_alignment
    })
    
    # Calculate geometric mean across factors, handling zeros and negatives
    final_factor = all_factors.apply(lambda x: np.sign(x.prod()) * np.power(np.abs(x.prod()), 1/4), axis=1)
    
    return final_factor
