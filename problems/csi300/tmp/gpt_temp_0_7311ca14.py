import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate required periods for lookback
    for i in range(6, len(df)):
        current_data = df.iloc[i]
        past_data = df.iloc[:i+1]  # All data up to current day
        
        # Primary Momentum Absorption
        close_t = current_data['close']
        close_t_5 = df.iloc[i-5]['close'] if i >= 5 else np.nan
        close_t_3 = df.iloc[i-3]['close'] if i >= 3 else np.nan
        close_t_6 = df.iloc[i-6]['close'] if i >= 6 else np.nan
        close_t_2 = df.iloc[i-2]['close'] if i >= 2 else np.nan
        close_t_4 = df.iloc[i-4]['close'] if i >= 4 else np.nan
        close_t_1 = df.iloc[i-1]['close'] if i >= 1 else np.nan
        
        volume_t = current_data['volume']
        amount_t = current_data['amount']
        
        # Primary Momentum Absorption
        if close_t > close_t_5 and not np.isnan(close_t_5):
            primary_momentum = (close_t - close_t_5) * volume_t / amount_t
        elif close_t < close_t_5 and not np.isnan(close_t_5):
            primary_momentum = (close_t_5 - close_t) * volume_t / amount_t
        else:
            primary_momentum = 0
        
        # Secondary Momentum Decay Profile
        if i >= 6 and not any(np.isnan([close_t, close_t_3, close_t_6])):
            momentum_decay = ((close_t/close_t_3 - 1) - (close_t_3/close_t_6 - 1)) * volume_t / amount_t
        else:
            momentum_decay = 0
        
        # Momentum Acceleration Efficiency
        if i >= 4 and not any(np.isnan([close_t, close_t_2, close_t_4])):
            momentum_accel = ((close_t/close_t_2 - 1) - (close_t_2/close_t_4 - 1)) * volume_t / amount_t
        else:
            momentum_accel = 0
        
        # Dynamic Price Anchors (4-day window)
        if i >= 4:
            window_data = df.iloc[i-4:i+1]
            volume_weighted_high_anchor = window_data['high'].max() * window_data['volume'].sum()
            volume_weighted_low_anchor = window_data['low'].min() * window_data['volume'].sum()
        else:
            volume_weighted_high_anchor = current_data['high'] * volume_t
            volume_weighted_low_anchor = current_data['low'] * volume_t
        
        # Current Session Anchor
        current_session_anchor = (current_data['high'] + current_data['low'] + current_data['close']) / 3 * volume_t
        
        # Anchor-Relative Position Signals
        anchor_range = volume_weighted_high_anchor - volume_weighted_low_anchor
        if anchor_range > 0:
            upper_position_pressure = (current_data['high'] - volume_weighted_high_anchor) / anchor_range
            lower_position_support = (volume_weighted_low_anchor - current_data['low']) / anchor_range
        else:
            upper_position_pressure = 0
            lower_position_support = 0
        
        price_range = current_data['high'] - current_data['low']
        if price_range > 0:
            session_anchor_deviation = (current_data['close'] - current_session_anchor) / price_range
        else:
            session_anchor_deviation = 0
        
        # Multi-Timeframe Order Flow
        if i >= 1:
            prev_range = df.iloc[i-1]['high'] - df.iloc[i-1]['low']
            if prev_range > 0:
                opening_flow_pressure = (current_data['open'] - df.iloc[i-1]['close']) / prev_range * volume_t
            else:
                opening_flow_pressure = 0
        else:
            opening_flow_pressure = 0
        
        if price_range > 0:
            closing_flow_momentum = (current_data['close'] - current_data['open']) / price_range * volume_t
        else:
            closing_flow_momentum = 0
        
        flow_persistence = (opening_flow_pressure + closing_flow_momentum) * volume_t / amount_t
        
        # Range Absorption Efficiency
        if current_data['close'] > current_data['open'] and price_range > 0:
            upward_range_efficiency = (current_data['high'] - current_data['open']) / price_range * volume_t
            downward_range_efficiency = 0
        elif current_data['close'] < current_data['open'] and price_range > 0:
            upward_range_efficiency = 0
            downward_range_efficiency = (current_data['open'] - current_data['low']) / price_range * volume_t
        else:
            upward_range_efficiency = 0
            downward_range_efficiency = 0
        
        # Volume Distribution Asymmetry
        if price_range > 0:
            upper_volume_concentration = (current_data['high'] - current_data['close']) / price_range * volume_t
            lower_volume_concentration = (current_data['close'] - current_data['low']) / price_range * volume_t
            volume_skew_ratio = (upper_volume_concentration - lower_volume_concentration) / volume_t if volume_t > 0 else 0
        else:
            upper_volume_concentration = 0
            lower_volume_concentration = 0
            volume_skew_ratio = 0
        
        # Price-Volume Cointegration
        current_cointegration = volume_t / price_range if price_range > 0 else 0
        
        if i >= 3:
            prev_range_3 = df.iloc[i-3]['high'] - df.iloc[i-3]['low']
            prev_volume_3 = df.iloc[i-3]['volume']
            if prev_range_3 > 0:
                cointegration_momentum = current_cointegration - (prev_volume_3 / prev_range_3)
            else:
                cointegration_momentum = 0
        else:
            cointegration_momentum = 0
        
        # Hierarchical Signal Integration
        # Momentum-Anchor Convergence
        momentum_anchor_convergence = (primary_momentum * (upper_position_pressure + lower_position_support) + 
                                     momentum_decay * session_anchor_deviation)
        
        # Flow-Efficiency Alignment
        range_efficiency = upward_range_efficiency + downward_range_efficiency
        flow_efficiency_alignment = (opening_flow_pressure + closing_flow_momentum) * range_efficiency + \
                                   flow_persistence * volume_skew_ratio
        
        # Multi-Scale Signal Weighting
        short_term_weight = primary_momentum * opening_flow_pressure
        medium_term_weight = momentum_accel * closing_flow_momentum
        persistence_weight = cointegration_momentum * flow_persistence
        
        multi_scale_weighting = short_term_weight + medium_term_weight + persistence_weight
        
        # Base Composite Factor
        base_composite = (momentum_anchor_convergence + flow_efficiency_alignment) * multi_scale_weighting
        
        # Volume-Weighted Smoothing (3-day window)
        if i >= 2:
            window_start = max(0, i-2)
            window_data = df.iloc[window_start:i+1]
            window_volumes = window_data['volume'].values
            window_factors = []
            
            for j in range(window_start, i+1):
                if j >= 6:  # Ensure we have enough data for calculation
                    window_factors.append(result.iloc[j])
            
            if len(window_factors) > 0 and sum(window_volumes[-len(window_factors):]) > 0:
                smoothed_factor = np.average(window_factors, weights=window_volumes[-len(window_factors):])
            else:
                smoothed_factor = base_composite
        else:
            smoothed_factor = base_composite
        
        # Final Alpha Output
        final_alpha = smoothed_factor * volume_skew_ratio
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
