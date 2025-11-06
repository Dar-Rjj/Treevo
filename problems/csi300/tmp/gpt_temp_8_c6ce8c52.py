import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Asymmetric Gap Efficiency Analysis
        # Directional Gap Efficiency
        open_t = current_data['open'].iloc[-1]
        close_t_1 = current_data['close'].iloc[-2] if i >= 1 else np.nan
        high_t = current_data['high'].iloc[-1]
        low_t = current_data['low'].iloc[-1]
        
        if pd.isna(close_t_1):
            continue
            
        upward_gap_efficiency = 0
        downward_gap_efficiency = 0
        
        if open_t > close_t_1:
            if (high_t - low_t) > 0:
                upward_gap_efficiency = (open_t - close_t_1) / (high_t - low_t)
        elif open_t < close_t_1:
            if (high_t - low_t) > 0:
                downward_gap_efficiency = (close_t_1 - open_t) / (high_t - low_t)
        
        gap_efficiency_asymmetry = upward_gap_efficiency - downward_gap_efficiency
        
        # Multi-Timeframe Gap Persistence
        close_t_3 = current_data['close'].iloc[-4] if i >= 3 else np.nan
        open_t_1 = current_data['open'].iloc[-2] if i >= 1 else np.nan
        close_t_2 = current_data['close'].iloc[-3] if i >= 2 else np.nan
        
        if pd.isna(close_t_3) or pd.isna(open_t_1) or pd.isna(close_t_2):
            continue
            
        gap_momentum = np.sign(open_t - close_t_1) * (current_data['close'].iloc[-1] - close_t_3)
        
        # Calculate range for t-2 to t
        high_range = current_data['high'].iloc[-3:].max()
        low_range = current_data['low'].iloc[-3:].min()
        range_t_2_t = high_range - low_range
        
        gap_acceleration = 0
        if range_t_2_t > 0:
            current_gap = open_t - close_t_1
            prev_gap = open_t_1 - close_t_2
            gap_acceleration = (current_gap - prev_gap) / range_t_2_t
        
        gap_regime_shift = np.sign(gap_momentum) * np.sign(gap_acceleration) if gap_acceleration != 0 else 0
        
        # Path-Enhanced Gap Analysis
        gap_path_alignment = 0
        if abs(open_t - close_t_1) > 0:
            gap_path_alignment = gap_efficiency_asymmetry * (high_t - low_t) / abs(open_t - close_t_1)
        
        morning_gap_efficiency = 0
        if open_t > close_t_1 and (open_t - close_t_1) > 0:
            morning_gap_efficiency = (high_t - open_t) / (open_t - close_t_1)
        
        gap_persistence_continuity = gap_momentum * (1 - abs(gap_acceleration))
        
        # Fractal Momentum Microstructure
        # Multi-Scale Momentum Divergence
        close_t_5 = current_data['close'].iloc[-6] if i >= 5 else np.nan
        close_t_20 = current_data['close'].iloc[-21] if i >= 20 else np.nan
        
        if pd.isna(close_t_5) or pd.isna(close_t_20):
            continue
            
        ultra_short_momentum = current_data['close'].iloc[-1] - close_t_3
        short_term_momentum = current_data['close'].iloc[-1] - close_t_5
        medium_term_momentum = current_data['close'].iloc[-1] - close_t_20
        
        fractal_momentum_divergence = (ultra_short_momentum - short_term_momentum) * (short_term_momentum - medium_term_momentum)
        
        # Price Path Efficiency Integration
        close_t = current_data['close'].iloc[-1]
        upward_path_efficiency = 0
        downward_path_efficiency = 0
        
        if close_t > open_t and (close_t - open_t) > 0:
            upward_path_efficiency = (high_t - open_t) / (close_t - open_t)
        elif close_t < open_t and (open_t - close_t) > 0:
            downward_path_efficiency = (open_t - low_t) / (open_t - close_t)
        
        path_efficiency_momentum = upward_path_efficiency - downward_path_efficiency
        
        # Microstructure-Momentum Synchronization
        gap_momentum_alignment = np.sign(open_t - close_t_1) * fractal_momentum_divergence
        path_momentum_convergence = path_efficiency_momentum * np.sign(fractal_momentum_divergence) if fractal_momentum_divergence != 0 else 0
        multi_timeframe_momentum_coherence = np.sign(ultra_short_momentum) * np.sign(short_term_momentum) * np.sign(medium_term_momentum)
        
        # Volume Microstructure Regime Detection
        # Volume Concentration Asymmetry
        amount_t = current_data['amount'].iloc[-1]
        volume_t = current_data['volume'].iloc[-1]
        volume_t_1 = current_data['volume'].iloc[-2] if i >= 1 else np.nan
        volume_t_2 = current_data['volume'].iloc[-3] if i >= 2 else np.nan
        
        if pd.isna(volume_t_1) or pd.isna(volume_t_2):
            continue
            
        volume_absorption_efficiency = 0
        if (amount_t / volume_t) > 0:
            volume_absorption_efficiency = (open_t - close_t_1) / (amount_t / volume_t)
        
        range_absorption_strength = 0
        if (volume_t / (amount_t / volume_t)) > 0:
            range_absorption_strength = (high_t - low_t) / (volume_t / (amount_t / volume_t))
        
        volume_timing_asymmetry = (volume_t / volume_t_1) - (volume_t_1 / volume_t_2)
        
        # Trade Size Distribution Patterns
        average_trade_size = amount_t / volume_t if volume_t > 0 else 0
        
        # Calculate trade size volatility (5-day std)
        if i >= 5:
            recent_amounts = current_data['amount'].iloc[-5:]
            recent_volumes = current_data['volume'].iloc[-5:]
            trade_sizes = recent_amounts / recent_volumes
            trade_size_volatility = trade_sizes.std()
        else:
            trade_size_volatility = 0
        
        trade_size_concentration = 0
        if volume_t > 0 and abs(close_t - close_t_1) > 0:
            trade_size_concentration = amount_t / (volume_t * abs(close_t - close_t_1))
        
        # Volume-Price Efficiency Divergence
        # Calculate average price movement
        if i >= 6:
            recent_price_moves = abs(current_data['close'].iloc[-6:-1].values - current_data['close'].iloc[-7:-2].values)
            avg_price_move = recent_price_moves.mean()
        else:
            avg_price_move = 0
        
        high_volume_low_movement = 0
        low_volume_high_movement = 0
        
        current_price_move = abs(close_t - close_t_1)
        if current_price_move < avg_price_move and current_price_move > 0:
            high_volume_low_movement = volume_t / current_price_move
        
        if current_price_move > avg_price_move and volume_t > 0:
            low_volume_high_movement = current_price_move / volume_t
        
        volume_efficiency_anomaly = high_volume_low_movement - low_volume_high_movement
        
        # Volatility-Fractal Regime Context
        # Multi-Scale Volatility Assessment
        if i >= 5:
            recent_highs = current_data['high'].iloc[-6:-1]
            recent_lows = current_data['low'].iloc[-6:-1]
            avg_range = (recent_highs - recent_lows).mean()
            recent_volumes = current_data['volume'].iloc[-6:-1]
            avg_volume = recent_volumes.mean()
        else:
            avg_range = 1
            avg_volume = volume_t
        
        price_volatility_fractality = (high_t - low_t) / avg_range if avg_range > 0 else 1
        volume_volatility_fractality = volume_t / avg_volume if avg_volume > 0 else 1
        gap_volatility_fractality = abs(open_t - close_t_1) / avg_range if avg_range > 0 else 1
        
        # Intraday Volatility Patterns
        opening_volatility = abs(open_t - close_t_1) / close_t_1 if close_t_1 > 0 else 0
        
        closing_volatility = 0
        if close_t < high_t and close_t > 0:
            closing_volatility = abs(close_t - high_t) / close_t
        elif close_t > low_t and close_t > 0:
            closing_volatility = abs(close_t - low_t) / close_t
        
        intraday_volatility_compression = 0
        if abs(open_t - close_t_1) > 0:
            intraday_volatility_compression = (high_t - low_t) / abs(open_t - close_t_1)
        
        # Regime-Based Signal Modulation
        volatility_confidence = 1 - gap_volatility_fractality
        
        range_stability_multiplier = 0
        if (high_t - low_t) > 0:
            range_position = (close_t - low_t) / (high_t - low_t)
            range_stability_multiplier = 1 / (1 + abs(range_position - 0.5))
        
        volume_regime_adjustment = volume_volatility_fractality * range_stability_multiplier
        
        # Asymmetric Composite Alpha Synthesis
        # Core Gap-Momentum Efficiency
        base_asymmetric_gap_momentum = gap_efficiency_asymmetry * fractal_momentum_divergence
        path_enhanced_gap_momentum = base_asymmetric_gap_momentum * path_efficiency_momentum
        volume_confirmed_gap_momentum = path_enhanced_gap_momentum * volume_efficiency_anomaly
        
        # Microstructure Regime Layer
        absorption_validated_momentum = volume_confirmed_gap_momentum * (1 + 0.2 * volume_absorption_efficiency)
        trade_size_adjustment = absorption_validated_momentum * trade_size_concentration
        volume_timing_enhanced = trade_size_adjustment * (1 + 0.1 * volume_timing_asymmetry)
        
        # Volatility-Regime Application
        range_stability_factor = volume_timing_enhanced * range_stability_multiplier
        volatility_confidence_weighted = range_stability_factor * volatility_confidence
        intraday_volatility_adjusted = volatility_confidence_weighted * intraday_volatility_compression
        
        # Final Multi-Dimensional Alpha
        asymmetric_gap_momentum_factor = intraday_volatility_adjusted
        
        result.iloc[i] = asymmetric_gap_momentum_factor
    
    return result
