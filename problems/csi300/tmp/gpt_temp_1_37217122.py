import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Gap-Efficiency Momentum Framework
    # Gap Persistence Efficiency
    gap_persistence = (data['open'] / data['close'].shift(1) - 1) * (data['close'] / data['open'] - 1)
    intraday_efficiency = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    gap_efficiency_interaction = gap_persistence * intraday_efficiency
    
    # Multi-Timeframe Momentum Divergence
    short_term_momentum = data['close'] / data['close'].shift(3) - 1
    medium_term_momentum = data['close'] / data['close'].shift(8) - 1
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Efficiency-Weighted Momentum
    efficiency_momentum = intraday_efficiency / (abs(data['close'] / data['close'].shift(1) - 1) + 0.0001)
    weighted_signal = momentum_divergence * efficiency_momentum
    
    # Volume-Pressure Confirmation System
    # Directional Volume Analysis
    directional_volume = np.sign(data['close'] - data['close'].shift(1)) * data['volume']
    volume_pressure = data['volume'] / (data['high'] - data['low']).replace(0, np.nan)
    volume_acceleration = (data['volume'] / data['volume'].shift(3) - 1) - (data['volume'] / data['volume'].shift(8) - 1)
    
    # Amount-Volume Integration
    amount_efficiency = data['amount'] / data['volume'].replace(0, np.nan)
    price_acceleration = (data['close'] - 2 * data['close'].shift(1) + data['close'].shift(2)) / (data['high'] - data['low']).replace(0, np.nan)
    volume_price_convergence = price_acceleration * amount_efficiency
    
    # Volume Flow Patterns
    def calculate_volume_flow(data, window):
        volume_flow = []
        for i in range(len(data)):
            if i < window:
                volume_flow.append(np.nan)
                continue
            window_data = data.iloc[i-window:i+1]
            up_volume = window_data['volume'][window_data['close'] > window_data['close'].shift(1)].sum()
            volume_flow.append(up_volume / window_data['volume'].sum() if window_data['volume'].sum() > 0 else np.nan)
        return pd.Series(volume_flow, index=data.index)
    
    volume_flow_ratio_short = calculate_volume_flow(data, 2)
    volume_flow_ratio_long = calculate_volume_flow(data, 7)
    volume_flow_ratio = volume_flow_ratio_short / volume_flow_ratio_long.replace(0, np.nan)
    
    volume_consistency = 1 / (data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()).replace(0, np.nan)
    
    # Range-Breakout Pressure Framework
    # Breakout Analysis
    def rolling_max_high(data, window):
        return data['high'].rolling(window=window).max().shift(1)
    
    breakout_distance = data['close'] - rolling_max_high(data, 5)
    intraday_recovery = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    breakout_efficiency = breakout_distance * intraday_recovery
    
    # Pressure Integration
    morning_momentum = (data['high'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    afternoon_momentum = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    session_divergence = morning_momentum - afternoon_momentum
    
    # Pressure-Weighted Breakout
    volume_pressure_breakout = breakout_efficiency * volume_pressure
    session_enhanced_breakout = volume_pressure_breakout * session_divergence
    
    # Regime-Adaptive Signal Construction
    # Volatility Regime Detection
    def true_range(data):
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift(1))
        tr3 = abs(data['low'] - data['close'].shift(1))
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    volatility_ratio = (data['high'] - data['low']).rolling(window=3).mean() / (data['high'] - data['low']).rolling(window=8).mean()
    true_range_vals = true_range(data)
    volatility_momentum = true_range_vals / true_range_vals.shift(5) - 1
    
    # Efficiency Regime Analysis
    def count_condition(series, condition, window):
        return series.rolling(window=window).apply(lambda x: sum(condition(x)), raw=False)
    
    efficiency_persistence = count_condition(intraday_efficiency, lambda x: x > 0.6, 5) - count_condition(intraday_efficiency, lambda x: x < 0.4, 5)
    efficiency_volatility = intraday_efficiency.rolling(window=5).std()
    
    # Adaptive Weighting
    volatility_scaled_signal = weighted_signal * volatility_ratio
    efficiency_enhanced_momentum = momentum_divergence * efficiency_persistence
    regime_adapted_breakout = session_enhanced_breakout * volatility_momentum
    
    # Composite Alpha Synthesis
    # Core Signal Integration
    gap_efficiency_core = gap_efficiency_interaction * weighted_signal
    volume_pressure_core = volume_price_convergence * volume_flow_ratio
    breakout_pressure_core = volume_pressure_breakout * session_divergence
    
    # Confirmation Framework
    volume_flow_multiplier = 1 + (volume_flow_ratio - 1) * np.sign(momentum_divergence)
    volatility_adaptation = volatility_scaled_signal * volatility_momentum
    efficiency_filtering = efficiency_enhanced_momentum * efficiency_volatility
    
    # Final Alpha Construction
    core_composite = gap_efficiency_core * volume_pressure_core * breakout_pressure_core
    regime_adapted_signal = core_composite * volatility_adaptation * efficiency_filtering
    final_alpha = regime_adapted_signal * volume_flow_multiplier * volume_consistency
    
    return final_alpha
