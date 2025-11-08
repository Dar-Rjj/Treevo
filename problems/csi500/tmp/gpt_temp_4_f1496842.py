import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate variables
    data['close_ret'] = data['close'].pct_change()
    data['intraday_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['volume_ratio_1'] = data['volume'] / data['volume'].shift(1)
    data['volume_ratio_3'] = data['volume'] / data['volume'].shift(3)
    
    # Volume consistency calculation
    volume_consistency = pd.Series(0, index=data.index)
    for i in range(2, len(data)):
        if i >= 2:
            count_above = sum(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] for j in range(3) if i-j-1 >= 0)
            sign_consistency = np.sign(data['volume'].iloc[i] - data['volume'].iloc[i-1]) * np.sign(data['volume'].iloc[i-1] - data['volume'].iloc[i-2])
            volume_consistency.iloc[i] = count_above * sign_consistency
    
    # Intraday Momentum Decay with Volume Persistence
    momentum_decay = (data['close_ret'] / data['intraday_range']).replace([np.inf, -np.inf], np.nan)
    volume_momentum = (data['volume_ratio_1'] + data['volume_ratio_3']) / 2
    base_decay = momentum_decay * volume_momentum
    persistence_adjustment = base_decay * volume_consistency
    decay_factor = persistence_adjustment * np.sign(data['close_ret'])
    
    # Volatility Regime Transition Detector
    short_term_vol = (data['high'] - data['low']) / data['close']
    medium_term_vol = (data['high'].shift(5) - data['low'].shift(5)) / data['close'].shift(5)
    regime_ratio = short_term_vol / medium_term_vol
    volume_surge = data['volume'] / data['volume'].shift(5)
    
    volume_persistence = pd.Series(0, index=data.index)
    for i in range(2, len(data)):
        if i >= 2:
            count_above = sum(data['volume'].iloc[i-j] > data['volume'].iloc[i-j-1] for j in range(3) if i-j-1 >= 0)
            volume_persistence.iloc[i] = count_above
    
    vol_vol_alignment = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(regime_ratio - 1)
    base_signal = regime_ratio * volume_surge
    confirmation_score = base_signal * volume_persistence
    directional_bias = confirmation_score * vol_vol_alignment
    
    regime_count = pd.Series(0, index=data.index)
    for i in range(5, len(data)):
        if i >= 5:
            count_above = sum(regime_ratio.iloc[i-j] > 1 for j in range(5))
            regime_count.iloc[i] = count_above
    
    transition_momentum = directional_bias * data['close_ret']
    regime_persistence = transition_momentum * regime_count
    regime_factor = regime_persistence * np.sign(regime_ratio - 1)
    
    # Liquidity-Adjusted Mean Reversion
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    deviation = (typical_price - typical_price.shift(5)) / typical_price.shift(5)
    range_position = (data['close'] - data['low']) / (data['high'] - data['low'])
    momentum_extension = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    
    daily_range = (data['high'] - data['low']) / typical_price
    volume_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    effective_spread = daily_range * volume_efficiency
    
    volume_ratio_5 = data['volume'] / data['volume'].shift(5)
    volume_clustering = volume_persistence  # Reuse from above
    liquidity_score = volume_ratio_5 * volume_clustering
    
    price_extreme = deviation * range_position * momentum_extension
    liquidity_adjustment = price_extreme * effective_spread
    reversion_factor = liquidity_adjustment * liquidity_score * np.sign(-deviation)
    
    # Order Flow Imbalance with Momentum Decay
    up_day_amount = data['amount'].where(data['close'] > data['open'], 0)
    down_day_amount = data['amount'].where(data['close'] < data['open'], 0)
    net_flow = (up_day_amount - down_day_amount) / (up_day_amount + down_day_amount).replace(0, np.nan)
    
    flow_change = net_flow - net_flow.shift(3)
    flow_acceleration = flow_change - (net_flow.shift(3) - net_flow.shift(6))
    flow_strength = abs(net_flow) * np.sign(flow_change)
    
    # Flow streak analysis
    streak_length = pd.Series(1, index=data.index)
    streak_intensity = pd.Series(0, index=data.index)
    
    for i in range(1, len(data)):
        if i > 0 and np.sign(net_flow.iloc[i]) == np.sign(net_flow.iloc[i-1]):
            streak_length.iloc[i] = streak_length.iloc[i-1] + 1
            streak_intensity.iloc[i] = streak_intensity.iloc[i-1] + net_flow.iloc[i]
        else:
            streak_length.iloc[i] = 1
            streak_intensity.iloc[i] = net_flow.iloc[i]
    
    avg_streak = streak_length.rolling(window=5, min_periods=1).mean()
    streak_duration_ratio = streak_length / avg_streak.replace(0, np.nan)
    
    flow_momentum_decay = flow_strength / streak_duration_ratio.replace(0, np.nan)
    volume_flow_corr = np.sign(data['volume'] - data['volume'].shift(1)) * np.sign(net_flow - net_flow.shift(1))
    decay_confirmation = flow_momentum_decay * volume_flow_corr
    
    base_flow = net_flow * flow_strength
    persistence_adj = base_flow * streak_intensity
    flow_factor = persistence_adj * decay_confirmation * np.sign(-flow_change)
    
    # Range Efficiency Momentum
    daily_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low'])
    efficiency_trend = daily_efficiency / daily_efficiency.shift(3)
    efficiency_level = daily_efficiency / ((daily_efficiency.shift(1) + daily_efficiency.shift(2) + daily_efficiency.shift(3)) / 3)
    
    volume_per_unit = data['volume'] / (data['high'] - data['low'])
    volume_efficiency_trend = volume_per_unit / volume_per_unit.shift(3)
    efficiency_alignment = np.sign(efficiency_trend) * np.sign(volume_efficiency_trend)
    
    base_efficiency = daily_efficiency * efficiency_trend
    volume_confirmation = base_efficiency * volume_efficiency_trend
    alignment_score = volume_confirmation * efficiency_alignment
    
    momentum_integration = alignment_score * data['close_ret']
    range_adjustment = momentum_integration * efficiency_level
    efficiency_factor = range_adjustment * np.sign(efficiency_trend)
    
    # Combine all factors with equal weights
    factors = [decay_factor, regime_factor, reversion_factor, flow_factor, efficiency_factor]
    valid_factors = [f.fillna(0) for f in factors]
    
    # Normalize and combine
    combined_factor = pd.Series(0, index=data.index)
    for f in valid_factors:
        if len(f) > 0:
            f_normalized = (f - f.mean()) / f.std() if f.std() != 0 else f
            combined_factor = combined_factor.add(f_normalized, fill_value=0)
    
    return combined_factor
