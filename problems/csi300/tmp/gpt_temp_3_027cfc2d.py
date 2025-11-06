import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(df):
    """
    Multi-Scale Regime-Adaptive Convergence with Price-Volume Cointegration
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Cross-Asset Momentum Convergence Framework (simplified - using rolling windows)
        # For demonstration, we'll use rolling windows instead of actual peer groups
        returns_1d = current_data['close'].pct_change(1)
        returns_5d = current_data['close'].pct_change(5)
        returns_15d = current_data['close'].pct_change(15)
        
        # Micro-momentum (1-day)
        micro_momentum = returns_1d.iloc[-1]
        
        # Meso-momentum (5-day) ranking (simplified)
        meso_rank = (returns_5d.rolling(10).rank().iloc[-1] - 5.5) / 5.5 if i >= 29 else 0
        
        # Macro-momentum (15-day) persistence
        macro_persistence = returns_15d.rolling(5).std().iloc[-1] if i >= 34 else 0
        
        # Momentum convergence score
        momentum_convergence = (micro_momentum * meso_rank) + (meso_rank * macro_persistence)
        
        # Cross-asset momentum acceleration
        momentum_acceleration = (micro_momentum - meso_rank) * (meso_rank - macro_persistence)
        
        # Price-Volume Cointegration Analysis
        window_data = current_data.iloc[-20:]
        
        # Price-Volume Regression for cointegration
        if len(window_data) >= 20:
            X = window_data['volume'].values.reshape(-1, 1)
            y = window_data['close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            residuals = y - model.predict(X)
            
            current_residual = residuals[-1]
            prev_residual = residuals[-2] if len(residuals) > 1 else 0
            residual_momentum = current_residual - prev_residual
            
            # Historical residual range
            hist_residuals = residuals[:-1]
            if len(hist_residuals) > 0:
                hist_range = np.max(hist_residuals) - np.min(hist_residuals)
                current_deviation = np.abs(current_residual - np.mean(hist_residuals))
                normalized_deviation = current_deviation / hist_range if hist_range > 0 else 0
            else:
                normalized_deviation = 0
            
            # Cointegration signal
            cointegration_signal = np.sign(residual_momentum) * (1 - normalized_deviation)
        else:
            cointegration_signal = 0
            momentum_acceleration = 0
        
        # Multi-Dimensional Regime Detection
        # Volatility Regime
        short_term_vol = current_data['close'].iloc[-5:].std() if i >= 24 else 0
        medium_term_vol = current_data['close'].iloc[-20:].std() if i >= 39 else 0
        volatility_regime = np.sign(short_term_vol - medium_term_vol) if short_term_vol > 0 and medium_term_vol > 0 else 0
        
        # Efficiency Regime
        if i >= 1:
            volume_efficiency = current_data['volume'].iloc[-1] / abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) if abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) > 0 else 0
            range_efficiency = abs(current_data['close'].iloc[-1] - current_data['open'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) if (current_data['high'].iloc[-1] - current_data['low'].iloc[-1]) > 0 else 0
            amount_efficiency = abs(current_data['close'].iloc[-1] - current_data['close'].iloc[-2]) / current_data['amount'].iloc[-1] if current_data['amount'].iloc[-1] > 0 else 0
            
            efficiency_regime = np.sign(volume_efficiency - range_efficiency) * np.sign(amount_efficiency)
        else:
            efficiency_regime = 0
        
        # Market State
        high_efficiency_trending = 1 if (efficiency_regime > 0) and (volatility_regime > 0) else 0
        low_efficiency_mean_reversion = 1 if (efficiency_regime < 0) and (volatility_regime < 0) else 0
        
        # Microstructure Pressure Dynamics
        if (current_data['close'].iloc[-1] - current_data['low'].iloc[-1]) > 0 and (current_data['high'].iloc[-1] - current_data['open'].iloc[-1]) > 0:
            upward_pressure = (current_data['high'].iloc[-1] - current_data['open'].iloc[-1]) / (current_data['close'].iloc[-1] - current_data['low'].iloc[-1])
            downward_pressure = (current_data['open'].iloc[-1] - current_data['low'].iloc[-1]) / (current_data['high'].iloc[-1] - current_data['close'].iloc[-1])
            pressure_ratio = upward_pressure / downward_pressure if downward_pressure > 0 else 1
        else:
            pressure_ratio = 1
        
        # Volume Concentration
        volume_window = current_data['volume'].iloc[-5:]
        if len(volume_window) >= 5:
            peak_concentration = np.max(volume_window) / np.mean(volume_window) if np.mean(volume_window) > 0 else 1
            volume_persistence = sum(volume_window.iloc[j] > volume_window.iloc[j-1] for j in range(1, min(4, len(volume_window)))) if len(volume_window) > 1 else 0
            concentration_signal = peak_concentration * volume_persistence
        else:
            concentration_signal = 1
        
        # Regime-Adaptive Signal Construction
        # Base signal
        base_signal = momentum_convergence * cointegration_signal
        
        # Volume enhanced
        volume_enhanced = base_signal * concentration_signal
        
        # Pressure adjusted
        pressure_adjusted = volume_enhanced * pressure_ratio
        
        # Regime-weighted signal
        if high_efficiency_trending:
            regime_weighted = pressure_adjusted * momentum_acceleration
        elif low_efficiency_mean_reversion:
            regime_weighted = pressure_adjusted * (1 - normalized_deviation) if 'normalized_deviation' in locals() else pressure_adjusted
        else:
            regime_weighted = pressure_adjusted
        
        # Cross-timeframe validation
        timeframe_alignment = np.sign(micro_momentum) + np.sign(meso_rank) + np.sign(macro_persistence)
        divergence_filter = (1 - normalized_deviation) if 'normalized_deviation' in locals() else 1
        
        # Final signal synthesis
        regime_adaptive_core = regime_weighted * timeframe_alignment * divergence_filter
        efficiency_boost = regime_adaptive_core * (1 + efficiency_regime)
        
        # Volume convergence (simplified)
        if i >= 5:
            stock_volume_momentum = current_data['volume'].iloc[-1] / np.mean(current_data['volume'].iloc[-5:-1]) if np.mean(current_data['volume'].iloc[-5:-1]) > 0 else 1
            volume_convergence = np.sign(stock_volume_momentum - 1)  # Simplified peer comparison
        else:
            volume_convergence = 1
        
        final_alpha = efficiency_boost * volume_convergence
        
        result.iloc[i] = final_alpha
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
