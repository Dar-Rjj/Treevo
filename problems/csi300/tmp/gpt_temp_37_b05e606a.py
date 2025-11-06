import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate rolling windows
    for i in range(len(df)):
        if i < 20:  # Need at least 20 periods for calculations
            alpha.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Volatility-Regime Adaptive Framework
        # Volatility Regime Classification
        high_4d = current_data['high'].iloc[i-4:i+1].max()
        low_4d = current_data['low'].iloc[i-4:i+1].min()
        high_9d = current_data['high'].iloc[i-9:i+1].max()
        low_9d = current_data['low'].iloc[i-9:i+1].min()
        
        volatility_ratio = (high_4d - low_4d) / (high_9d - low_9d) if (high_9d - low_9d) > 0 else 1.0
        
        if volatility_ratio > 1.2:
            regime_weight = 1.5
        elif volatility_ratio < 0.8:
            regime_weight = 0.7
        else:
            regime_weight = 1.0
        
        # Asymmetric Volatility Analysis
        current_close = current_data['close'].iloc[i]
        upside_vol = (high_4d - current_close) / (high_4d - low_4d) if (high_4d - low_4d) > 0 else 0.5
        downside_vol = (current_close - low_4d) / (high_4d - low_4d) if (high_4d - low_4d) > 0 else 0.5
        
        # Asymmetric Mean Reversion Core
        # Multi-Timeframe Price Deviation
        close_5d = current_data['close'].iloc[i-4:i+1].mean()
        close_10d = current_data['close'].iloc[i-9:i+1].mean()
        close_20d = current_data['close'].iloc[i-19:i+1].mean()
        
        short_term_dev = (current_close - close_5d) / close_5d if close_5d > 0 else 0
        medium_term_dev = (current_close - close_10d) / close_10d if close_10d > 0 else 0
        long_term_dev = (current_close - close_20d) / close_20d if close_20d > 0 else 0
        
        # Asymmetric Reversion Momentum
        close_prev = current_data['close'].iloc[i-1]
        close_3d = current_data['close'].iloc[i-3]
        close_5d_ago = current_data['close'].iloc[i-5]
        
        ret_3d = (current_close / close_3d - 1) if close_3d > 0 else 0
        ret_5d = (current_close / close_5d_ago - 1) if close_5d_ago > 0 else 0
        
        if current_close > close_prev:
            upside_reversion_pressure = ret_3d - ret_5d
            downside_reversion_pressure = 0
        else:
            upside_reversion_pressure = 0
            downside_reversion_pressure = ret_3d - ret_5d
        
        reversion_asymmetry_ratio = (upside_reversion_pressure / downside_reversion_pressure 
                                   if downside_reversion_pressure != 0 else 1.0)
        
        # Microstructure Reversion Signals
        open_today = current_data['open'].iloc[i]
        close_prev = current_data['close'].iloc[i-1]
        high_today = current_data['high'].iloc[i]
        low_today = current_data['low'].iloc[i]
        
        opening_gap_reversion = (open_today - close_prev) / close_prev if close_prev > 0 else 0
        intraday_range_reversion = (current_close - open_today) / (high_today - low_today) if (high_today - low_today) > 0 else 0
        closing_pressure_reversion = (current_close - (high_today + low_today)/2) / ((high_today - low_today)/2) if (high_today - low_today) > 0 else 0
        
        # Volume-Efficiency Convergence Filter
        # Asymmetric Volume Analysis
        volume_today = current_data['volume'].iloc[i]
        amount_today = current_data['amount'].iloc[i]
        
        bull_volume_efficiency = (current_close - low_today) * volume_today / amount_today if amount_today > 0 else 0
        bear_volume_efficiency = (high_today - current_close) * volume_today / amount_today if amount_today > 0 else 0
        
        volume_efficiency_asymmetry = (bull_volume_efficiency / bear_volume_efficiency 
                                     if bear_volume_efficiency > 0 else 1.0)
        
        # Volume Stability Assessment
        volume_10d = current_data['volume'].iloc[i-9:i+1]
        volume_stability = volume_10d.std() / volume_10d.mean() if volume_10d.mean() > 0 else 1.0
        trade_efficiency = amount_today / (high_today - low_today) if (high_today - low_today) > 0 else 0
        
        # Multi-Timeframe Volume Convergence
        volume_5d_avg = current_data['volume'].iloc[i-4:i+1].mean()
        volume_10d_avg = current_data['volume'].iloc[i-9:i+1].mean()
        volume_20d_avg = current_data['volume'].iloc[i-19:i+1].mean()
        
        short_term_volume_trend = volume_today / volume_5d_avg if volume_5d_avg > 0 else 1.0
        medium_term_volume_structure = volume_10d_avg / volume_20d_avg if volume_20d_avg > 0 else 1.0
        
        # Volume Profile Consistency (simplified correlation)
        volume_10d_window = current_data['volume'].iloc[i-9:i+1].values
        price_change_10d = np.abs(current_data['close'].iloc[i-9:i+1].diff().dropna().values)
        if len(price_change_10d) == len(volume_10d_window) - 1:
            volume_profile_consistency = np.corrcoef(volume_10d_window[:-1], price_change_10d)[0,1] if len(price_change_10d) > 1 else 0
        else:
            volume_profile_consistency = 0
        
        # Asymmetric Momentum Integration
        # Directional Acceleration Components
        close_prev1 = current_data['close'].iloc[i-1]
        close_prev2 = current_data['close'].iloc[i-2]
        close_prev3 = current_data['close'].iloc[i-3]
        
        upside_acceleration = (current_close - close_prev1) - (close_prev1 - close_prev2)
        downside_acceleration = (close_prev1 - close_prev2) - (close_prev2 - close_prev3)
        
        acceleration_asymmetry = (upside_acceleration / downside_acceleration 
                                if downside_acceleration != 0 else 1.0)
        
        # Volume-Weighted Momentum
        volume_prev = current_data['volume'].iloc[i-1]
        high_prev = current_data['high'].iloc[i-1]
        low_prev = current_data['low'].iloc[i-1]
        close_prev = current_data['close'].iloc[i-1]
        
        bull_momentum_volume = ((current_close - low_today) * volume_today / 
                               ((close_prev - low_prev) * volume_prev)) if ((close_prev - low_prev) * volume_prev) > 0 else 1.0
        bear_momentum_volume = ((high_today - current_close) * volume_today / 
                               ((high_prev - close_prev) * volume_prev)) if ((high_prev - close_prev) * volume_prev) > 0 else 1.0
        
        momentum_volume_bias = bull_momentum_volume / bear_momentum_volume if bear_momentum_volume > 0 else 1.0
        
        # Pressure Acceleration Signals
        open_prev = current_data['open'].iloc[i-1]
        close_prev2 = current_data['close'].iloc[i-2]
        
        opening_pressure_change = (open_today - close_prev) - (open_prev - close_prev2)
        closing_pressure_change = (current_close - open_today) - (close_prev - open_prev)
        
        pressure_acceleration_asymmetry = (opening_pressure_change / closing_pressure_change 
                                         if closing_pressure_change != 0 else 1.0)
        
        # Composite Alpha Construction
        # Weighted Asymmetric Reversion
        volatility_weighted_deviation = -medium_term_dev * regime_weight
        asymmetric_reversion_score = volatility_weighted_deviation * reversion_asymmetry_ratio
        microstructure_adjustment = asymmetric_reversion_score * closing_pressure_reversion
        
        # Volume-Efficiency Convergence Score
        efficiency_quality = volume_efficiency_asymmetry * volume_stability
        volume_convergence = short_term_volume_trend * volume_profile_consistency
        convergence_strength = efficiency_quality * volume_convergence
        
        # Momentum Alignment Component
        acceleration_bias = acceleration_asymmetry * momentum_volume_bias
        pressure_alignment = pressure_acceleration_asymmetry * trade_efficiency
        momentum_quality = acceleration_bias * pressure_alignment
        
        # Final Alpha Output
        core_reversion_signal = microstructure_adjustment * convergence_strength
        momentum_confirmation = core_reversion_signal * momentum_quality
        final_alpha = momentum_confirmation * volume_efficiency_asymmetry
        
        alpha.iloc[i] = final_alpha
    
    return alpha
