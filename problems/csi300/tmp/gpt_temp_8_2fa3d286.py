import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper functions for entropy calculations
    def directional_entropy(series, window=5):
        returns = series.pct_change()
        direction = np.sign(returns)
        return direction.rolling(window=window).apply(lambda x: (x == 1).sum() / len(x) if len(x) == window else np.nan)
    
    def amplitude_entropy(high, low, window=5):
        ranges = high - low
        normalized_ranges = ranges / ranges.rolling(window=window).mean()
        return normalized_ranges.rolling(window=window).std()
    
    def gap_entropy(open_price, close_prev, window=5):
        gaps = (open_price - close_prev.shift(1)) / close_prev.shift(1)
        return gaps.rolling(window=window).std()
    
    def volume_pattern_complexity(volume, window=5):
        volume_changes = volume.pct_change()
        return volume_changes.rolling(window=window).std()
    
    def trade_size_distribution(amount, volume, window=5):
        avg_trade_size = amount / volume
        return avg_trade_size.rolling(window=window).std()
    
    def volume_volatility_coupling(volume, high, low, window=5):
        volatility = high - low
        volume_vol_corr = volume.rolling(window=window).corr(volatility)
        return volume_vol_corr
    
    def volatility_clustering(high, low, window=10):
        volatility = high - low
        vol_changes = volatility.pct_change()
        return vol_changes.rolling(window=window).std()
    
    # Asymmetric Information Velocity Dynamics
    # Multi-Scale Rejection-Entropy Patterns
    high_rejection = data['high'] - np.maximum(data['open'], data['close'])
    low_rejection = np.minimum(data['open'], data['close']) - data['low']
    rejection_asymmetry = high_rejection / (low_rejection + 1e-8)
    dir_entropy = directional_entropy(data['close'])
    entropy_weighted_rejection = rejection_asymmetry * dir_entropy
    
    # 3-day High Rejection Entropy
    high_3day = data['high'].rolling(window=3).max()
    close_max_3day = data['close'].rolling(window=3).max()
    high_rejection_3day = (data['high'] - close_max_3day) / (data['high'] - data['low'] + 1e-8)
    amp_entropy = amplitude_entropy(data['high'], data['low'])
    high_rejection_entropy_3day = high_rejection_3day * amp_entropy
    
    # 3-day Low Rejection Entropy
    close_min_3day = data['close'].rolling(window=3).min()
    low_rejection_3day = (close_min_3day - data['low']) / (data['high'] - data['low'] + 1e-8)
    gap_ent = gap_entropy(data['open'], data['close'])
    low_rejection_entropy_3day = low_rejection_3day * gap_ent
    
    # Net Rejection-Entropy Momentum
    net_rejection_entropy_momentum = (high_rejection_entropy_3day - low_rejection_entropy_3day) * np.sign(data['close'] - data['open'])
    
    # Efficiency-Entropy Integration
    intraday_efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    vol_pattern_comp = volume_pattern_complexity(data['volume'])
    intraday_efficiency_entropy = intraday_efficiency * vol_pattern_comp
    
    # Medium-term Efficiency Entropy
    close_5day_change = data['close'] - data['close'].shift(5)
    high_5day_max = data['high'].rolling(window=5).max()
    low_5day_min = data['low'].rolling(window=5).min()
    medium_term_efficiency = close_5day_change / (high_5day_max - low_5day_min + 1e-8)
    trade_size_dist = trade_size_distribution(data['amount'], data['volume'])
    medium_term_efficiency_entropy = medium_term_efficiency * trade_size_dist
    
    # Efficiency-Entropy Divergence
    efficiency_entropy_divergence = intraday_efficiency_entropy - medium_term_efficiency_entropy
    
    # Efficiency-Entropy Momentum
    efficiency_entropy_momentum = intraday_efficiency_entropy / intraday_efficiency_entropy.shift(1) - 1
    
    # Rejection-Efficiency-Entropy Alignment
    entropy_enhanced_rejection = net_rejection_entropy_momentum * intraday_efficiency_entropy
    
    # Nonlinear Order Flow Velocity
    # Entropy-Enhanced Order Flow Dynamics
    bid_pressure = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    bid_pressure_entropy = bid_pressure * data['volume'] * dir_entropy
    
    ask_pressure = (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)
    ask_pressure_entropy = ask_pressure * data['volume'] * amp_entropy
    
    order_flow_entropy_imbalance = (bid_pressure_entropy - ask_pressure_entropy) / (data['volume'] + 1e-8)
    
    order_flow_entropy_momentum = order_flow_entropy_imbalance - order_flow_entropy_imbalance.rolling(window=5).mean()
    
    # Volume-Liquidity Entropy Velocity
    liquidity_absorption = data['volume'] / (data['high'] - data['low'] + 1e-8)
    liquidity_absorption_entropy = liquidity_absorption * vol_pattern_comp
    
    absorption_entropy_momentum = liquidity_absorption_entropy / liquidity_absorption_entropy.rolling(window=5).mean()
    
    volume_efficiency_entropy = data['volume'] * intraday_efficiency * trade_size_dist
    
    # Volatility-Entropy Breakout Patterns
    # Multi-Scale Volatility-Entropy Dynamics
    volatility_expansion = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    volume_change = data['volume'] / data['volume'].shift(1) - 1
    vol_clustering = volatility_clustering(data['high'], data['low'])
    volatility_expansion_entropy = volatility_expansion * volume_change * vol_clustering
    
    # Gap Behavior Entropy Integration
    gap_fill = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)
    gap_threshold = 0.02 * data['close'].shift(1)
    gap_fill_entropy_momentum = gap_fill * (np.abs(data['open'] - data['close'].shift(1)) > gap_threshold)
    
    # Breakout-Flow-Entropy Integration
    upside_breakout = data['high'] / data['high'].shift(1) - 1
    downside_breakout = data['low'] / data['low'].shift(1) - 1
    breakout_entropy_asymmetry = upside_breakout - downside_breakout
    
    flow_enhanced_breakout_entropy = breakout_entropy_asymmetry * order_flow_entropy_imbalance
    
    # Entropy-Velocity Convergence Validation
    # Multi-Dimensional Entropy Alignment
    efficiency_flow_entropy_divergence = np.sign(efficiency_entropy_momentum) * np.sign(order_flow_entropy_momentum)
    rejection_absorption_alignment = np.sign(net_rejection_entropy_momentum) * np.sign(absorption_entropy_momentum)
    
    # Entropy Persistence Patterns
    def calculate_persistence(series, window=3):
        signs = np.sign(series)
        persistence = signs.rolling(window=window).apply(
            lambda x: (x == x.shift(1)).sum() / (window - 1) if len(x) == window else np.nan
        )
        return persistence
    
    order_flow_persistence = calculate_persistence(order_flow_entropy_imbalance)
    efficiency_consistency = calculate_persistence(intraday_efficiency_entropy.diff())
    absorption_persistence = calculate_persistence(liquidity_absorption_entropy.diff())
    rejection_persistence = calculate_persistence(net_rejection_entropy_momentum)
    
    # Entropy-Velocity Alpha Synthesis
    # Core Entropy-Velocity Components
    rejection_flow_entropy_velocity = entropy_enhanced_rejection * order_flow_entropy_momentum
    absorption_entropy_momentum_velocity = absorption_entropy_momentum * (data['close'] / data['close'].shift(1) - 1)
    
    # Institutional Activity Entropy
    avg_trade_size = data['amount'] / (data['volume'] + 1e-8)
    institutional_activity = avg_trade_size.rolling(window=5).apply(
        lambda x: (x > x.mean()).sum() / len(x) if len(x) == 5 else np.nan
    )
    institutional_activity_entropy = institutional_activity * amp_entropy
    
    trade_size_efficiency_entropy = avg_trade_size * intraday_efficiency_entropy
    institutional_efficiency_entropy_velocity = institutional_activity_entropy * trade_size_efficiency_entropy * intraday_efficiency_entropy
    
    breakout_flow_entropy_velocity = flow_enhanced_breakout_entropy * volatility_expansion_entropy
    
    # Entropy-Enhanced Divergence Signals
    microstructure_confirmed_entropy_velocity = rejection_flow_entropy_velocity * efficiency_flow_entropy_divergence
    absorption_aligned_entropy_momentum = absorption_entropy_momentum_velocity * rejection_absorption_alignment
    
    trade_size_divergence = np.sign(avg_trade_size.pct_change()) * np.sign(order_flow_entropy_momentum)
    institutional_flow_entropy_momentum = institutional_efficiency_entropy_velocity * trade_size_divergence
    
    price_flow_divergence = np.sign(data['close'].pct_change()) * np.sign(order_flow_entropy_imbalance)
    validated_breakout_entropy_velocity = breakout_flow_entropy_velocity * price_flow_divergence
    
    # Entropy-Weighted Persistence Factors
    flow_entropy_persistence_weight = order_flow_persistence * efficiency_consistency
    absorption_entropy_persistence_weight = absorption_persistence * rejection_persistence
    
    gap_volume_intensity = data['volume'] / data['volume'].rolling(window=5).mean()
    gap_threshold_2 = 0.015 * data['close'].shift(1)
    gap_volume_entropy_intensity = gap_volume_intensity * (np.abs(data['open'] - data['close'].shift(1)) > gap_threshold_2)
    gap_behavior_entropy_adjustment = gap_fill_entropy_momentum * gap_volume_entropy_intensity
    
    range_expansion_signal = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8) > 1.2
    range_entropy_dynamics = range_expansion_signal.astype(float) * breakout_entropy_asymmetry
    
    # Final Entropy-Velocity Alpha
    primary_factor = microstructure_confirmed_entropy_velocity * flow_entropy_persistence_weight
    secondary_factor = absorption_aligned_entropy_momentum * absorption_entropy_persistence_weight
    tertiary_factor = institutional_flow_entropy_momentum * gap_behavior_entropy_adjustment
    quaternary_factor = validated_breakout_entropy_velocity * range_entropy_dynamics
    
    # Composite Entropy-Velocity Alpha
    composite_alpha = primary_factor * secondary_factor * tertiary_factor * quaternary_factor
    
    return composite_alpha
