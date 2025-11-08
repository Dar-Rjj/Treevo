import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Multi-Timeframe Momentum with Volume-Price Regime Alignment alpha factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic price components
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['daily_range'] = data['high'] / data['low'] - 1
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Multi-period returns
    data['two_day_return'] = data['close'] / data['close'].shift(1) - 1
    data['five_day_return'] = data['close'] / data['close'].shift(4) - 1
    data['ten_day_return'] = data['close'] / data['close'].shift(9) - 1
    data['twenty_day_return'] = data['close'] / data['close'].shift(19) - 1
    
    # Acceleration components
    data['short_term_acceleration'] = data['five_day_return'] - data['two_day_return']
    data['medium_term_acceleration'] = data['twenty_day_return'] - data['ten_day_return']
    
    # Gap analysis
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_fill'] = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    data['gap_direction_alignment'] = np.sign(data['overnight_gap']) * np.sign(data['intraday_return'])
    
    # Trend consistency
    returns_signs = pd.DataFrame({
        'intraday': np.sign(data['intraday_return']),
        'five_day': np.sign(data['five_day_return']),
        'twenty_day': np.sign(data['twenty_day_return'])
    })
    data['multi_timeframe_sign_agreement'] = returns_signs.apply(
        lambda x: (x == x['intraday']).sum() if not pd.isna(x['intraday']) else 0, axis=1
    )
    data['acceleration_concordance'] = np.sign(data['short_term_acceleration']) * np.sign(data['medium_term_acceleration'])
    
    # Volume dynamics
    data['daily_volume_change'] = data['volume'] / data['volume'].shift(1) - 1
    data['short_term_volume_ratio'] = data['volume'] / data['volume'].shift(4)
    
    # Volume persistence calculation
    data['volume_increase'] = (data['volume'] > data['volume'].shift(1)).astype(int)
    data['volume_persistence'] = data['volume_increase'].rolling(window=5, min_periods=1).apply(
        lambda x: len(x) if len(x) == 0 else (x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)).max(), 
        raw=False
    )
    
    # Volume profile
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=5, min_periods=1).sum()
    data['volume_trend_strength'] = (data['volume'] - data['volume'].shift(4)) / (data['volume'].shift(4) + 1e-8)
    data['volume_stability'] = 1 / (abs(data['daily_volume_change']) + 0.001)
    
    # Price-volume alignment
    data['daily_alignment'] = np.sign(data['intraday_return']) * np.sign(data['daily_volume_change'])
    data['multi_day_alignment'] = np.sign(data['five_day_return']) * np.sign(data['volume_trend_strength'])
    
    # Alignment persistence
    data['positive_alignment'] = (data['daily_alignment'] > 0).astype(int)
    data['alignment_persistence'] = data['positive_alignment'].rolling(window=5, min_periods=1).apply(
        lambda x: len(x) if len(x) == 0 else (x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1)).max(), 
        raw=False
    )
    
    # Strength alignment
    data['magnitude_correlation'] = abs(data['intraday_return']) * abs(data['daily_volume_change'])
    data['range_volume_ratio'] = data['daily_range'] / (abs(data['daily_volume_change']) + 0.001)
    data['efficiency_ratio'] = abs(data['intraday_return']) / (data['daily_range'] + 0.001)
    
    # Regime classification
    data['short_term_volatility'] = data['daily_range'].rolling(window=5, min_periods=1).mean()
    data['medium_term_volatility'] = data['daily_range'].rolling(window=20, min_periods=1).mean()
    data['volatility_ratio'] = data['short_term_volatility'] / (data['medium_term_volatility'] + 1e-8)
    
    data['volume_level'] = data['volume'] / data['volume'].rolling(window=20, min_periods=1).mean().shift(1)
    
    # Momentum regime scoring
    data['sign_agreement_score'] = data['multi_timeframe_sign_agreement'] / 3
    data['magnitude_score'] = (abs(data['intraday_return']) + abs(data['five_day_return']) + abs(data['twenty_day_return'])) / 3
    data['acceleration_score'] = (data['short_term_acceleration'] + data['medium_term_acceleration']) / 2
    
    # Core momentum signal calculation
    data['core_momentum'] = (
        (4 * data['intraday_return']) + 
        (2 * data['five_day_return']) + 
        (1 * data['twenty_day_return'])
    ) / 7
    
    # Acceleration enhancement
    data['acceleration_boost'] = (0.1 * data['short_term_acceleration']) + (0.05 * data['medium_term_acceleration'])
    data['core_momentum'] = data['core_momentum'] + data['acceleration_boost']
    
    # Pattern recognition bonuses
    gap_fill_bonus = 0.2 * data['gap_fill'] * (abs(data['gap_fill']) < 1)
    close_position_bonus = 0.1 * (data['close_position'] - 0.5)
    efficiency_bonus = 0.15 * data['efficiency_ratio']
    
    data['core_momentum'] = data['core_momentum'] + gap_fill_bonus + close_position_bonus + efficiency_bonus
    
    # Volume-price confirmation
    daily_alignment_multiplier = 1 + 0.3 * np.sign(data['daily_alignment'])
    persistence_multiplier = 1 + 0.1 * data['alignment_persistence']
    multi_day_alignment_bonus = 0.2 * np.sign(data['multi_day_alignment'])
    
    volume_magnitude_scaling = 1 + 0.5 * abs(data['daily_volume_change'])
    efficiency_scaling = 1 + 0.3 * data['efficiency_ratio']
    range_volume_adjustment = 1 / (1 + data['range_volume_ratio'])
    
    # Volume regime adaptation
    volume_level_scaling = np.where(
        data['volume_level'] > 1.2, 1.4,
        np.where(data['volume_level'] < 0.8, 0.7, 1.0)
    )
    
    volume_trend_scaling = np.where(
        data['volume_trend_strength'] > 0.2, 1.3,
        np.where(data['volume_trend_strength'] < -0.2, 0.8, 1.0)
    )
    
    persistence_scaling = np.where(
        data['volume_persistence'] >= 3, 1.2,
        np.where(data['volume_persistence'] <= 1, 0.8, 1.0)
    )
    
    # Combine volume-price confirmation components
    directional_confirmation = daily_alignment_multiplier * persistence_multiplier + multi_day_alignment_bonus
    strength_confirmation = volume_magnitude_scaling * efficiency_scaling * range_volume_adjustment
    volume_regime_adaptation = volume_level_scaling * volume_trend_scaling * persistence_scaling
    
    data['volume_price_confirmation'] = directional_confirmation * strength_confirmation * volume_regime_adaptation
    
    # Regime-adaptive refinement
    volatility_adjustment = np.where(
        data['volatility_ratio'] > 1.3, 0.6,
        np.where(data['volatility_ratio'] > 1.1, 0.8,
        np.where(data['volatility_ratio'] < 0.7, 1.5,
        np.where(data['volatility_ratio'] < 0.9, 1.2, 1.0)))
    )
    
    momentum_regime_filter = np.where(
        (data['sign_agreement_score'] == 1) & (data['acceleration_score'] > 0), 1.8,
        np.where(data['sign_agreement_score'] >= 0.67, 1.2,
        np.where(data['sign_agreement_score'] == 0, 0.3, 0.8))
    )
    
    # Signal validation
    min_volume_requirement = data['daily_volume_change'] > -0.5
    persistence_requirement = data['alignment_persistence'] >= 1
    consistency_check = np.sign(data['core_momentum']) == np.sign(data['intraday_return'])
    volatility_floor = data['daily_range'] > 0.001
    
    valid_signal = min_volume_requirement & persistence_requirement & consistency_check & volatility_floor
    
    # Final composite alpha calculation
    base_composite = data['core_momentum'] * data['volume_price_confirmation']
    regime_adjusted = base_composite * volatility_adjustment * momentum_regime_filter
    
    # Apply validation filtering
    alpha = regime_adjusted.where(valid_signal, 0)
    
    return alpha
