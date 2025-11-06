import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Multi-Scale Fractal Momentum with Asymmetric Decay
        # Short-Term Asymmetric Momentum (3-day)
        if i >= 3:
            close_t = current_data['close'].iloc[i]
            close_t3 = current_data['close'].iloc[i-3]
            open_t = current_data['open'].iloc[i]
            high_t = current_data['high'].iloc[i]
            low_t = current_data['low'].iloc[i]
            volume_t = current_data['volume'].iloc[i]
            
            # Asymmetric Price Momentum
            price_momentum = (close_t - close_t3) / close_t3
            direction = (close_t - open_t) / (abs(close_t - open_t) + 1e-8)
            asymmetric_price_momentum = price_momentum * direction
            
            # Range-Adjusted Asymmetric Momentum
            high_range = current_data['high'].iloc[i-3:i+1].max()
            low_range = current_data['low'].iloc[i-3:i+1].min()
            range_adjusted_momentum = (close_t - close_t3) / (high_range - low_range + 1e-8) * volume_t
            
            # Fractal Opening Pressure
            if close_t > open_t:
                fractal_opening_pressure = (high_t - open_t) / (close_t - open_t + 1e-8)
            else:
                fractal_opening_pressure = 0
            
            # Asymmetric Fractal Efficiency
            close_t1 = current_data['close'].iloc[i-1]
            fractal_efficiency = abs(close_t - close_t1) / (high_t - low_t + 1e-8) * np.sign(close_t - open_t)
            
            # Medium-Term Divergence with Volume Asymmetry (5-day vs 10-day)
            if i >= 10:
                close_t5 = current_data['close'].iloc[i-5]
                close_t10 = current_data['close'].iloc[i-10]
                high_t5 = current_data['high'].iloc[i-5]
                low_t5 = current_data['low'].iloc[i-5]
                high_t10 = current_data['high'].iloc[i-10]
                low_t10 = current_data['low'].iloc[i-10]
                volume_t1 = current_data['volume'].iloc[i-1]
                
                # Volume-Weighted Fractal Divergence
                divergence_5day = (close_t - close_t5) / (high_t5 - low_t5 + 1e-8)
                divergence_10day = (close_t - close_t10) / (high_t10 - low_t10 + 1e-8)
                volume_weighted_divergence = (divergence_5day - divergence_10day) * volume_t
                
                # Asymmetric Momentum Persistence
                open_low_diff = open_t - low_t
                high_open_diff = high_t - open_t
                max_diff = max(open_low_diff, high_open_diff)
                momentum_persistence = (close_t - open_t) / (max_diff + 1e-8) * volume_t / (volume_t1 + 1e-8)
                
                # Combined Asymmetric Fractal Decay
                asymmetric_cross_fractal = (range_adjusted_momentum * volume_weighted_divergence * 
                                          fractal_efficiency * momentum_persistence)
            else:
                asymmetric_cross_fractal = 0
                volume_weighted_divergence = 0
                momentum_persistence = 0
        else:
            asymmetric_cross_fractal = 0
            fractal_opening_pressure = 0
            fractal_efficiency = 0
        
        # Microstructure Pressure Dynamics with Fractal Volume Patterns
        if i >= 1:
            close_t1 = current_data['close'].iloc[i-1]
            amount_t = current_data['amount'].iloc[i]
            volume_t1 = current_data['volume'].iloc[i-1]
            
            # Asymmetric Price Pressure Components
            # Opening Regime Pressure
            opening_regime_pressure = ((open_t - close_t1) / (close_t1 + 1e-8)) * (volume_t / (amount_t + 1e-8))
            
            # Intraday Asymmetric Pressure
            intraday_pressure = (((close_t - low_t) / (high_t - low_t + 1e-8)) - 
                               ((high_t - close_t) / (high_t - low_t + 1e-8))) * volume_t
            
            # Fractal Reversal Intensity
            fractal_reversal = (abs(close_t - (high_t + low_t)/2) / (high_t - low_t + 1e-8) * 
                              np.sign(close_t - open_t))
            
            # Asymmetric Gap Reversion
            asymmetric_gap = ((open_t / (close_t1 + 1e-8) - 1) * (close_t / (open_t + 1e-8) - 1) * volume_t)
            
            # Fractal Volume-Pressure Regimes
            # Volume Acceleration Regime
            volume_acceleration = (volume_t / (volume_t1 + 1e-8)) * np.sign(close_t - open_t) * (amount_t / (volume_t + 1e-8))
            
            # Price-Volume Association Regime
            price_volume_association = (np.sign(close_t - close_t1) * 
                                      np.sign(volume_t - volume_t1) * volume_t)
            
            # Volume Burst Regime Detection
            price_change_pct = abs(close_t - close_t1) / (close_t1 + 1e-8)
            if price_change_pct > 0.02:
                volume_burst = volume_t / (volume_t1 + 1e-8)
            else:
                volume_burst = 1
            
            # Fractal Volume Echo
            if i >= 2:
                volume_t2 = current_data['volume'].iloc[i-2]
                price_dir_current = np.sign(close_t - close_t1)
                price_dir_prev = np.sign(close_t1 - current_data['close'].iloc[i-2])
                if price_dir_current != price_dir_prev:
                    volume_echo = volume_t / (volume_t2 + 1e-8) * amount_t
                else:
                    volume_echo = 1
            else:
                volume_echo = 1
            
            # Fractal Volume Efficiency
            volume_efficiency = (amount_t / (volume_t * (high_t - low_t + 1e-8)) * 
                               np.sign(close_t - open_t))
            
            # Microstructure Fractal Composite
            regime_pressure_volume = (intraday_pressure * volume_acceleration * 
                                    price_volume_association * volume_echo)
        else:
            regime_pressure_volume = 0
            opening_regime_pressure = 0
            fractal_reversal = 0
            asymmetric_gap = 0
            volume_burst = 1
            volume_efficiency = 0
        
        # Multi-Dimensional Regime Transitions with Fractal Validation
        if i >= 1:
            # Fractal Volatility Efficiency Detection
            # True Range Regime
            true_range = max(high_t - low_t, 
                           abs(high_t - close_t1), 
                           abs(low_t - close_t1))
            true_range_regime = true_range * volume_t
            
            # Price Efficiency Regime
            price_efficiency = abs(close_t - close_t1) / (true_range + 1e-8) * amount_t
            
            # Volatility Jump Regime
            if i >= 2:
                high_t1 = current_data['high'].iloc[i-1]
                low_t1 = current_data['low'].iloc[i-1]
                open_t1 = current_data['open'].iloc[i-1]
                volume_t1 = current_data['volume'].iloc[i-1]
                volatility_jump = ((high_t - low_t) / (open_t + 1e-8)) / ((high_t1 - low_t1) / (open_t1 + 1e-8)) * volume_t
            else:
                volatility_jump = 1
            
            # Asymmetric Trending Indicator
            asymmetric_trending = (2 * price_efficiency - 1) * np.sign(close_t - open_t)
            
            # Multi-Timeframe Regime Coherence
            if i >= 3:
                close_t2 = current_data['close'].iloc[i-2]
                close_t3 = current_data['close'].iloc[i-3]
                asymmetric_short_term = (np.sign(close_t - close_t2) * 
                                       np.sign(close_t1 - close_t3) * volume_t)
                
                # Fractal Compression Detection
                if i >= 2:
                    high_t1 = current_data['high'].iloc[i-1]
                    low_t1 = current_data['low'].iloc[i-1]
                    fractal_compression = ((high_t - low_t) / (high_t1 - low_t1 + 1e-8)) * (volume_t / (volume_t1 + 1e-8))
                else:
                    fractal_compression = 1
                
                # Regime Persistence with Volume (simplified)
                regime_persistence = volume_t
            else:
                asymmetric_short_term = 0
                fractal_compression = 1
                regime_persistence = volume_t
            
            # Multi-Dimensional Regime Signal
            multi_dimensional_signal = (asymmetric_trending * asymmetric_short_term * 
                                      fractal_compression * regime_persistence)
        else:
            multi_dimensional_signal = 0
            asymmetric_trending = 0
        
        # Asymmetric Alpha Construction with Fractal Validation
        # Core Asymmetric Momentum Factor
        core_momentum = asymmetric_cross_fractal * regime_pressure_volume
        
        # Fractal Volume Regime Validation
        volume_regime_validation = core_momentum * volume_efficiency
        
        # Multi-Dimensional Regime Enhancement
        multi_dimensional_enhancement = volume_regime_validation * multi_dimensional_signal
        
        # Final Asymmetric Alpha Synthesis
        # Primary Alpha Component
        primary_alpha = (multi_dimensional_enhancement * asymmetric_gap * fractal_reversal)
        
        # Secondary Validation
        secondary_validation = primary_alpha * volume_burst
        
        # Final Asymmetric Alpha
        final_alpha = (secondary_validation * opening_regime_pressure * fractal_opening_pressure)
        
        alpha.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
