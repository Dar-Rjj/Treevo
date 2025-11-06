import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate required lags
    for i in range(5, len(df)):
        if i < 5:  # Need at least 5 days of data
            continue
            
        # Extract current and historical data
        data = df.iloc[max(0, i-5):i+1]
        
        # Price Fractal Components
        close_t = data['close'].iloc[-1]
        close_t1 = data['close'].iloc[-2] if len(data) > 1 else close_t
        close_t2 = data['close'].iloc[-3] if len(data) > 2 else close_t1
        close_t3 = data['close'].iloc[-4] if len(data) > 3 else close_t2
        
        # Nonlinear Momentum
        if len(data) >= 4:
            nm1 = (close_t - close_t2)**2 / (close_t2 + 0.001)
            nm2 = (close_t1 - close_t3)**2 / (close_t3 + 0.001)
            nonlinear_momentum = nm1 - nm2
        else:
            nonlinear_momentum = 0
            
        # Volatility Fractal
        if len(data) >= 6:
            vol_t = (data['high'].iloc[-1] - data['low'].iloc[-1])**1.5
            vol_t5 = (data['high'].iloc[-6] - data['low'].iloc[-6])**1.5
            volatility_fractal = (vol_t / (vol_t5 + 1e-8)) - 1
        else:
            volatility_fractal = 0
            
        # Asymmetric Gap Momentum
        if len(data) >= 2:
            gap = (data['open'].iloc[-1] - data['close'].iloc[-2])
            range_t = data['high'].iloc[-1] - data['low'].iloc[-1]
            asymmetric_gap_momentum = gap * range_t / (data['close'].iloc[-2] + 1e-8)
        else:
            asymmetric_gap_momentum = 0
            
        # Volume Fractal Dynamics
        volume_t = data['volume'].iloc[-1]
        volume_t1 = data['volume'].iloc[-2] if len(data) > 1 else volume_t
        volume_t2 = data['volume'].iloc[-3] if len(data) > 2 else volume_t1
        volume_t3 = data['volume'].iloc[-4] if len(data) > 3 else volume_t2
        
        # Volume Power Law
        if len(data) >= 4:
            vpl_t = volume_t**0.7
            vpl_t3 = volume_t3**0.7
            volume_power_law = (vpl_t / (vpl_t3 + 1e-8)) - 1
        else:
            volume_power_law = 0
            
        # Volume-Volatility Coupling
        if len(data) >= 1:
            range_t = data['high'].iloc[-1] - data['low'].iloc[-1]
            volume_volatility_coupling = volume_t * (range_t**0.5)
        else:
            volume_volatility_coupling = 0
            
        # Nonlinear Volume Acceleration
        if len(data) >= 3:
            nva1 = (volume_t / (volume_t1 + 1e-8))**2
            nva2 = (volume_t1 / (volume_t2 + 1e-8))**2
            nonlinear_volume_acceleration = nva1 - nva2
        else:
            nonlinear_volume_acceleration = 0
            
        # Cross-Fractal Interactions
        price_volume_fractal = nonlinear_momentum * volume_power_law
        volatility_volume_fractal = volatility_fractal * volume_volatility_coupling
        gap_volume_fractal = asymmetric_gap_momentum * nonlinear_volume_acceleration
        
        # Regime Detection
        momentum_regime = np.sign(nonlinear_momentum) * np.sign(price_volume_fractal)
        
        if volatility_fractal > 0.1:
            volatility_regime = "High"
        elif volatility_fractal < -0.1:
            volatility_regime = "Low"
        else:
            volatility_regime = "Stable"
            
        if volume_power_law > 0.05:
            volume_regime = "Expanding"
        elif volume_power_law < -0.05:
            volume_regime = "Contracting"
        else:
            volume_regime = "Neutral"
            
        # Regime Transition Signals
        momentum_transition = abs(nonlinear_momentum) * (1 + abs(price_volume_fractal))
        volatility_transition = volatility_fractal * volume_volatility_coupling
        volume_transition = volume_power_law * nonlinear_volume_acceleration
        
        # Cross-Regime Interactions
        momentum_volatility_interaction = momentum_transition * volatility_transition
        momentum_volume_interaction = momentum_transition * volume_transition
        triple_regime_signal = momentum_volatility_interaction * volume_transition
        
        # Nonlinear Alpha Components
        # Power Law Components
        if len(data) >= 2:
            price_power = (close_t / (close_t1 + 1e-8))**1.3 - 1
            volume_power = (volume_t / (volume_t1 + 1e-8))**0.8 - 1
            range_t = data['high'].iloc[-1] - data['low'].iloc[-1]
            range_t1 = data['high'].iloc[-2] - data['low'].iloc[-2]
            range_power = (range_t**0.9 / (range_t1**0.9 + 1e-8)) - 1
        else:
            price_power = volume_power = range_power = 0
            
        # Cross-Power Interactions
        price_volume_power = price_power * volume_power
        price_range_power = price_power * range_power
        volume_range_power = volume_power * range_power
        
        # Nonlinear Momentum Integration
        power_momentum = price_volume_power * price_range_power
        fractal_power = volatility_volume_fractal * volume_range_power
        regime_power = triple_regime_signal * power_momentum
        
        # Asymmetric Pressure Dynamics
        # Pressure Components
        if len(data) >= 1:
            high_t = data['high'].iloc[-1]
            low_t = data['low'].iloc[-1]
            buy_pressure = ((close_t - low_t)**1.2) / ((high_t - low_t)**1.2 + 1e-8)
            sell_pressure = ((high_t - close_t)**1.2) / ((high_t - low_t)**1.2 + 1e-8)
            pressure_asymmetry = buy_pressure - sell_pressure
        else:
            buy_pressure = sell_pressure = pressure_asymmetry = 0
            
        # Multi-day Pressure
        if len(data) >= 2:
            # Calculate previous day pressures
            close_t1 = data['close'].iloc[-2]
            high_t1 = data['high'].iloc[-2]
            low_t1 = data['low'].iloc[-2]
            buy_pressure_t1 = ((close_t1 - low_t1)**1.2) / ((high_t1 - low_t1)**1.2 + 1e-8)
            sell_pressure_t1 = ((high_t1 - close_t1)**1.2) / ((high_t1 - low_t1)**1.2 + 1e-8)
            
            two_day_buy_pressure = buy_pressure + buy_pressure_t1
            two_day_sell_pressure = sell_pressure + sell_pressure_t1
            pressure_momentum = two_day_buy_pressure - two_day_sell_pressure
        else:
            pressure_momentum = 0
            
        # Pressure-Fractal Integration
        fractal_buy_pressure = buy_pressure * volatility_fractal
        fractal_sell_pressure = sell_pressure * volume_power_law
        net_fractal_pressure = fractal_buy_pressure - fractal_sell_pressure
        
        # Final Alpha Construction
        # Core Alpha Components
        regime_power_core = regime_power * pressure_momentum
        fractal_pressure_core = net_fractal_pressure * fractal_power
        nonlinear_core = regime_power_core * fractal_pressure_core
        
        # Regime-Weighted Alpha
        if volatility_regime == "High":
            regime_weighted_alpha = nonlinear_core * (1 + volatility_fractal)
        elif volume_regime == "Expanding":
            regime_weighted_alpha = nonlinear_core * (1 + volume_power_law)
        else:
            regime_weighted_alpha = nonlinear_core * (1 + abs(momentum_transition))
        
        # Final Alpha Factor
        pressure_enhanced_alpha = regime_weighted_alpha * (1 + pressure_asymmetry)
        
        # Cross-Fractal Interactions aggregation
        cross_fractal_interactions = (price_volume_fractal + volatility_volume_fractal + gap_volume_fractal) / 3
        
        fractal_confirmed_alpha = pressure_enhanced_alpha * cross_fractal_interactions
        
        alpha.iloc[i] = fractal_confirmed_alpha
    
    # Fill NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
