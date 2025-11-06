import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Efficiency-Momentum with Regime Persistence & Breakout Confirmation
    """
    data = df.copy()
    
    # Multi-Timeframe Efficiency Assessment
    # Ultra-Short Efficiency (2-day)
    data['ultra_short_true_range'] = data['high'].rolling(window=2).max() - data['low'].rolling(window=2).min()
    data['ultra_short_price_movement'] = abs(data['close'] - data['close'].shift(2))
    data['ultra_short_efficiency'] = data['ultra_short_price_movement'] / data['ultra_short_true_range'].replace(0, np.nan)
    
    # Short-Term Efficiency (5-day)
    data['short_term_true_range'] = data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()
    data['short_term_price_movement'] = abs(data['close'] - data['close'].shift(5))
    data['short_term_efficiency'] = data['short_term_price_movement'] / data['short_term_true_range'].replace(0, np.nan)
    
    # Efficiency Acceleration Patterns
    data['ultra_short_acceleration'] = data['ultra_short_efficiency'] - data['short_term_efficiency']
    data['efficiency_alignment'] = np.sign(data['ultra_short_efficiency']) * np.sign(data['short_term_efficiency'])
    data['efficiency_momentum'] = data['ultra_short_efficiency'] * data['short_term_efficiency']
    
    # Momentum Acceleration & Regime Detection
    # Multi-Timeframe Momentum
    data['momentum_short'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_long'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum Acceleration
    data['momentum_acceleration'] = (data['momentum_short'] - data['momentum_medium']) / abs(data['momentum_medium']).replace(0, np.nan)
    
    # Price Regime Persistence
    returns = data['close'].pct_change()
    data['daily_return_sign'] = np.sign(returns)
    
    # Directional persistence
    directional_persistence = []
    current_streak = 0
    for sign in data['daily_return_sign']:
        if sign == 0:
            current_streak = 0
        elif len(directional_persistence) == 0:
            current_streak = 1
        elif sign == directional_persistence[-1]:
            current_streak += 1
        else:
            current_streak = 1
        directional_persistence.append(current_streak)
    data['directional_persistence'] = directional_persistence
    
    # Magnitude consistency
    abs_returns_5d = abs(returns).rolling(window=5)
    data['magnitude_consistency'] = abs_returns_5d.std() / abs_returns_5d.mean().replace(0, np.nan)
    
    # Volume-Price Efficiency System
    # Volume Concentration Analysis
    data['volume_ratio_5d'] = data['volume'] / data['volume'].shift(5).replace(0, np.nan)
    
    # Volume During Moves
    large_move_mask = abs(data['close'] - data['close'].shift(1)) > 0.02 * data['close'].shift(1)
    data['volume_during_moves'] = np.where(large_move_mask, 
                                         data['volume'] / data['volume'].shift(1).replace(0, np.nan), 
                                         np.nan)
    
    # Volume Persistence
    volume_diff_1 = data['volume'] - data['volume'].shift(1)
    volume_diff_2 = data['volume'].shift(1) - data['volume'].shift(2)
    data['volume_persistence'] = np.sign(volume_diff_1) * np.sign(volume_diff_2)
    
    # Volume-Price Efficiency Metrics
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Asymmetry (5-day)
    volume_up = 0
    volume_down = 0
    volume_asymmetry = []
    for i in range(len(data)):
        if i >= 5:
            up_vol = sum(data['volume'].iloc[i-4:i+1].values * 
                        (data['close'].iloc[i-4:i+1] > data['close'].iloc[i-5:i].values))
            down_vol = sum(data['volume'].iloc[i-4:i+1].values * 
                          (data['close'].iloc[i-4:i+1] < data['close'].iloc[i-5:i].values))
            volume_asymmetry.append(up_vol / down_vol if down_vol != 0 else np.nan)
        else:
            volume_asymmetry.append(np.nan)
    data['volume_asymmetry'] = volume_asymmetry
    
    # Volume-Price Alignment (5-day)
    volume_price_alignment = []
    for i in range(len(data)):
        if i >= 5:
            returns_window = data['close'].iloc[i-4:i+1].values / data['close'].iloc[i-5:i].values - 1
            volume_window = data['volume'].iloc[i-4:i+1].values
            numerator = sum(volume_window * returns_window)
            denominator = sum(returns_window)
            volume_price_alignment.append(numerator / denominator if denominator != 0 else np.nan)
        else:
            volume_price_alignment.append(np.nan)
    data['volume_price_alignment'] = volume_price_alignment
    
    # Volume Regime Characteristics
    # Volume trend persistence (10-day regression slope)
    data['volume_trend_persistence'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan, raw=False
    )
    
    # Volume volatility
    volume_changes = data['volume'].pct_change()
    data['volume_volatility'] = volume_changes.rolling(window=5).std() / abs(volume_changes.rolling(window=5).mean()).replace(0, np.nan)
    
    # Position and Breakout Context
    # Intraday Position Strength
    daily_range = data['high'] - data['low']
    data['daily_range_position'] = (data['close'] - data['low']) / daily_range.replace(0, np.nan)
    data['position_momentum'] = data['daily_range_position'] - (
        (data['close'].shift(1) - data['low'].shift(1)) / 
        (data['high'].shift(1) - data['low'].shift(1)).replace(0, np.nan)
    )
    
    # Key Level Proximity
    data['resistance_distance'] = (data['high'].rolling(window=5).max() - data['close']) / data['close']
    data['support_distance'] = (data['close'] - data['low'].rolling(window=5).min()) / data['close']
    data['level_break_potential'] = np.sign(data['support_distance'] - data['resistance_distance']) * data['ultra_short_acceleration']
    
    # Breakout Validation
    data['position_breakout_alignment'] = data['position_momentum'] * np.sign(data['close'] - data['close'].shift(1))
    data['efficiency_breakout_confirmation'] = data['efficiency_momentum'] * (
        (data['high'] > data['high'].shift(1)) | (data['low'] < data['low'].shift(1))
    ).astype(float)
    data['volume_breakout_strength'] = data['volume_asymmetry'] * np.sign(data['close'] - data['close'].shift(1))
    
    # Composite Factor Generation
    # Core Efficiency-Momentum Signal
    base_signal = data['efficiency_momentum'] * data['momentum_acceleration']
    acceleration_component = data['ultra_short_acceleration'] * data['volume_trend_persistence']
    alignment_score = data['efficiency_alignment'] * np.sign(data['momentum_acceleration'])
    
    # Regime & Breakout Enhancement
    regime_strength = data['directional_persistence'] * (1 - data['magnitude_consistency'])
    volume_regime_quality = data['volume_trend_persistence'] / data['volume_volatility'].replace(0, np.nan)
    breakout_efficiency = data['efficiency_breakout_confirmation'] * data['volume_breakout_strength']
    
    # Contextual Adjustment
    position_context = base_signal * (1 + data['position_breakout_alignment'] * 0.1)
    level_break_adjustment = base_signal * (1 + data['level_break_potential'] * 0.15)
    regime_validation = base_signal * (1 + regime_strength * 0.2)
    
    # Final Composite Factor
    composite_factor = (
        base_signal * 0.4 +
        acceleration_component * 0.2 +
        alignment_score * 0.1 +
        regime_strength * 0.1 +
        breakout_efficiency * 0.2
    ) * volume_regime_quality.fillna(1)
    
    # Apply contextual adjustments
    final_factor = (
        position_context * 0.3 +
        level_break_adjustment * 0.3 +
        regime_validation * 0.4
    )
    
    return final_factor
