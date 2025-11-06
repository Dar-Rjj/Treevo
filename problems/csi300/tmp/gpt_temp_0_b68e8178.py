import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-scale Divergence Elasticity
    # Short-term (3-day)
    price_ret_3d = data['close'] / data['close'].shift(3) - 1
    vol_ret_3d = data['volume'] / data['volume'].shift(3) - 1
    range_ratio_3d = (data['high'] - data['low']) / (data['close'] - data['open'])
    short_term_div_elasticity = (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan) * range_ratio_3d
    
    # Medium-term (8-day)
    price_ret_8d = data['close'] / data['close'].shift(8) - 1
    vol_ret_8d = data['volume'] / data['volume'].shift(8) - 1
    range_ratio_5d = (data['high'].shift(5) - data['low'].shift(5)) / (data['close'].shift(5) - data['open'].shift(5))
    medium_term_div_elasticity = (price_ret_8d / vol_ret_8d).replace([np.inf, -np.inf], np.nan) * range_ratio_5d
    
    # Long-term (13-day)
    price_ret_13d = data['close'] / data['close'].shift(13) - 1
    vol_ret_13d = data['volume'] / data['volume'].shift(13) - 1
    range_ratio_10d = (data['high'].shift(10) - data['low'].shift(10)) / (data['close'].shift(10) - data['open'].shift(10))
    long_term_div_elasticity = (price_ret_13d / vol_ret_13d).replace([np.inf, -np.inf], np.nan) * range_ratio_10d
    
    # Elasticity Rejection Divergence
    upper_shadow = (data['high'] - data['close']) / (data['high'] - data['low'])
    lower_shadow = (data['close'] - data['low']) / (data['high'] - data['low'])
    upper_shadow_divergence = upper_shadow * (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan)
    lower_shadow_divergence = lower_shadow * (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan)
    net_divergence_rejection = lower_shadow_divergence - upper_shadow_divergence
    
    # Divergence-Elasticity Momentum
    divergence_acceleration = short_term_div_elasticity - medium_term_div_elasticity
    elasticity_persistence_divergence = medium_term_div_elasticity - long_term_div_elasticity
    rejection_divergence_alignment = net_divergence_rejection * divergence_acceleration
    
    # Amount-Pressure Dynamics
    # Trade Size Pressure Analysis (simplified - using amount/volume as proxy for trade size)
    avg_trade_size = data['amount'] / data['volume']
    trade_size_volatility = avg_trade_size.rolling(window=5).std()
    large_trade_pressure = (avg_trade_size / avg_trade_size.rolling(window=10).mean()) * (data['close'] - data['open']) / (data['high'] - data['low'])
    small_trade_dispersion_pressure = (1 / (avg_trade_size / avg_trade_size.rolling(window=10).mean())) * (data['close'] - data['low']) / (data['high'] - data['low'])
    trade_size_volatility_pressure = trade_size_volatility * (data['high'] - data['low']) / data['amount']
    
    # Depth Efficiency Pressure
    pressure_impact_efficiency = ((data['high'] - data['low']) / data['amount']) * ((data['close'] - data['open']) / (data['high'] - data['low']))
    amount_pressure_ratio = (data['amount'] / (data['high'] - data['low'])) * ((data['close'] - data['low']) / (data['high'] - data['low']))
    
    # Calculate correlation between amount and volume over 10 days
    amount_volume_corr = data['amount'].rolling(window=10).corr(data['volume'])
    depth_pressure_stability = amount_volume_corr * ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Pressure-Momentum Integration
    high_depth_pressure_momentum = (data['amount'] > data['amount'].shift(5)).astype(float) * ((data['close'] - data['open']) / (data['high'] - data['low']))
    low_depth_pressure_momentum = (data['amount'] <= data['amount'].shift(5)).astype(float) * ((data['close'] - data['low']) / (data['high'] - data['low']))
    pressure_divergence_from_elasticity = (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan) - ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Volume-Elasticity Flow Patterns
    # Transaction Elasticity Density
    elasticity_density = (data['volume'] / (data['high'] - data['low'])) * ((data['high'] - data['low']) / (data['close'] - data['open']))
    density_momentum_elasticity = ((data['volume'] / (data['high'] - data['low'])) / (data['volume'].shift(2) / (data['high'].shift(2) - data['low'].shift(2)))) * ((data['high'] - data['low']) / (data['close'] - data['open']))
    
    # Calculate average volume density over past 8 days
    vol_density_8d = (data['volume'].shift(1) / (data['high'].shift(1) - data['low'].shift(1))).rolling(window=8).mean()
    flow_concentration_elasticity = ((data['volume'] / (data['high'] - data['low'])) / vol_density_8d) * ((data['high'] - data['low']) / (data['close'] - data['open']))
    
    # Price Impact Elasticity Flow
    elasticity_impact_efficiency = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * ((data['high'] - data['low']) / (data['close'] - data['open']))
    flow_impact_elasticity = (data['volume'] / (data['high'] - data['low'])) * (abs(data['close'] - data['open']) / (data['high'] - data['low']))
    pressure_elasticity_flow = ((data['close'] - data['open']) / (data['high'] - data['low'])) * (data['volume'] / (data['high'] - data['low']))
    
    # Bid-Ask Elasticity Dynamics
    intraday_elasticity_pressure = ((data['close'] - data['open']) / (data['high'] - data['low'])) * ((data['high'] - data['close']) / (data['high'] - data['low']))
    pressure_elasticity_flow_divergence = ((data['close'] - data['open']) / (data['high'] - data['low'])) - (data['volume'] / (data['high'] - data['low']))
    flow_pressure_elasticity = (data['volume'] / (data['high'] - data['low'])) * ((data['close'] - data['low']) / (data['high'] - data['low']))
    
    # Divergence-Elasticity Breakout Detection
    # Multi-scale Breakout Patterns
    # Calculate max divergence over past 20 days
    div_20d = pd.Series(index=data.index, dtype=float)
    for i in range(20, len(data)):
        start_idx = i - 20
        end_idx = i - 1
        div_values = []
        for j in range(start_idx, end_idx):
            price_ret_temp = data['close'].iloc[j] / data['close'].iloc[j-3] - 1
            vol_ret_temp = data['volume'].iloc[j] / data['volume'].iloc[j-3] - 1
            div_val = (price_ret_temp / vol_ret_temp) if vol_ret_temp != 0 else 0
            div_values.append(div_val)
        div_20d.iloc[i] = max(div_values) if div_values else 0
    
    divergence_breakout = ((price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan) / div_20d).replace([np.inf, -np.inf], np.nan)
    
    # Volume-Elasticity Efficiency Divergence
    volume_weighted_divergence = (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan) * data['volume']
    elasticity_volatility_divergence = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))) * (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan)
    efficiency_divergence_score = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) - ((data['volume'] / data['volume'].shift(5)) * (price_ret_3d / vol_ret_3d).replace([np.inf, -np.inf], np.nan))
    co_movement_divergence = volume_weighted_divergence * efficiency_divergence_score
    
    # Regime Transition Divergence
    divergence_regime_shift = ((price_ret_3d / vol_ret_3d + price_ret_8d / vol_ret_8d) / (price_ret_13d / vol_ret_13d)).replace([np.inf, -np.inf], np.nan)
    volume_confirmation_divergence = (data['volume'] / data['volume'].shift(1)) * (data['volume'] / data['volume'].shift(3)) * divergence_breakout
    flow_elasticity_divergence = (data['volume'] / (data['high'] - data['low'])) * divergence_acceleration
    pressure_regime_divergence = ((data['close'] - data['open']) / (data['high'] - data['low'])) * divergence_regime_shift
    
    # Core Component Construction
    divergence_component = net_divergence_rejection * divergence_acceleration * flow_elasticity_divergence
    pressure_component = pressure_impact_efficiency * amount_pressure_ratio * flow_pressure_elasticity
    elasticity_component = elasticity_impact_efficiency * density_momentum_elasticity * pressure_elasticity_flow
    breakout_component = co_movement_divergence * volume_confirmation_divergence * pressure_regime_divergence
    
    # Dynamic Regime Weighting
    vol_condition = data['volume'] > data['volume'].shift(5)
    div_condition = (price_ret_3d / vol_ret_3d) > (price_ret_8d / vol_ret_8d)
    high_pressure_expansion = vol_condition & div_condition.replace([np.inf, -np.inf], False)
    
    low_pressure_compression = (~vol_condition) & (~div_condition.replace([np.inf, -np.inf], True))
    
    breakout_vol_condition = (data['volume'] / data['volume'].shift(1)) * (data['volume'] / data['volume'].shift(3)) > 2
    breakout_transition = (divergence_breakout > 1.5) | breakout_vol_condition
    
    # Initialize weights
    w_div = pd.Series(0.3, index=data.index)
    w_pressure = pd.Series(0.3, index=data.index)
    w_elasticity = pd.Series(0.2, index=data.index)
    w_breakout = pd.Series(0.2, index=data.index)
    
    # Apply regime-specific weights
    w_div[high_pressure_expansion] = 0.4
    w_pressure[high_pressure_expansion] = 0.3
    w_elasticity[high_pressure_expansion] = 0.2
    w_breakout[high_pressure_expansion] = 0.1
    
    w_div[low_pressure_compression] = 0.25
    w_pressure[low_pressure_compression] = 0.35
    w_elasticity[low_pressure_compression] = 0.25
    w_breakout[low_pressure_compression] = 0.15
    
    w_div[breakout_transition] = 0.3
    w_pressure[breakout_transition] = 0.25
    w_elasticity[breakout_transition] = 0.2
    w_breakout[breakout_transition] = 0.25
    
    # Final Alpha Generation
    weighted_component_sum = (divergence_component * w_div + 
                            pressure_component * w_pressure + 
                            elasticity_component * w_elasticity + 
                            breakout_component * w_breakout)
    
    # Apply enhancements and adjustments
    volume_divergence_enhanced = weighted_component_sum * volume_weighted_divergence
    pressure_efficiency_adjusted = volume_divergence_enhanced * pressure_impact_efficiency
    elasticity_confirmed = pressure_efficiency_adjusted + (intraday_elasticity_pressure * net_divergence_rejection)
    
    # Final alpha factor
    alpha_factor = elasticity_confirmed
    
    # Clean infinite values and replace with NaN
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    
    return alpha_factor
