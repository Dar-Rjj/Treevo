import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize factor series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        current_data = df.iloc[:i+1].copy()
        
        # Price Fractal Dimension (10-day)
        if i >= 10:
            fractal_window = current_data.iloc[-10:]
            high_low_range = (fractal_window['high'] - fractal_window['low']) / fractal_window['close']
            price_changes = fractal_window['close'].pct_change().dropna()
            
            if len(price_changes) > 1:
                # Hurst exponent approximation
                lags = range(1, min(6, len(price_changes)))
                tau = []
                for lag in lags:
                    if len(price_changes) >= lag:
                        tau.append(np.std(np.diff(price_changes, lag)))
                
                if len(tau) > 1:
                    hurst = np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0]
                    fractal_dim = 2 - hurst
                else:
                    fractal_dim = 1.5
            else:
                fractal_dim = 1.5
        else:
            fractal_dim = 1.5
        
        # Multi-Scale Volatility Adjustment
        current_close = current_data.iloc[-1]['close']
        current_range = (current_data.iloc[-1]['high'] - current_data.iloc[-1]['low']) / current_data.iloc[-1]['close']
        
        momentum_short = 0
        momentum_medium = 0
        momentum_long = 0
        
        if i >= 5:
            close_5 = current_data.iloc[-6]['close']
            momentum_short = (current_close - close_5) / max(current_range, 0.001)
        
        if i >= 10:
            close_10 = current_data.iloc[-11]['close']
            momentum_medium = (current_close - close_10) / max(current_range, 0.001)
        
        if i >= 20:
            close_20 = current_data.iloc[-21]['close']
            momentum_long = (current_close - close_20) / max(current_range, 0.001)
        
        fractal_momentum = (momentum_short + momentum_medium + momentum_long) / 3 * fractal_dim
        
        # Volume-Confluence Efficiency
        if i >= 5:
            recent_5 = current_data.iloc[-5:]
            
            # Price Range Utilization
            high_close_ratio = (recent_5['high'] - recent_5['close']) / (recent_5['high'] - recent_5['low'] + 1e-8)
            close_low_ratio = (recent_5['close'] - recent_5['low']) / (recent_5['high'] - recent_5['low'] + 1e-8)
            
            directional_efficiency = np.mean(np.abs(high_close_ratio - close_low_ratio))
            
            # 5-day range utilization
            range_utilization = np.mean((recent_5['close'] - recent_5['low']) / (recent_5['high'] - recent_5['low'] + 1e-8))
            
            # Volume Distribution Analysis
            volume_concentration = np.mean(recent_5['volume'] * (recent_5['high'] - recent_5['low']) / recent_5['close'])
            
            # Volume-volatility correlation (6-day)
            if i >= 6:
                vol_window = current_data.iloc[-6:]
                volume_vol_corr = vol_window['volume'].corr(vol_window['high'] - vol_window['low'])
                if pd.isna(volume_vol_corr):
                    volume_vol_corr = 0
            else:
                volume_vol_corr = 0
            
            volume_efficiency = (directional_efficiency + range_utilization + volume_concentration) / 3 * (1 + volume_vol_corr)
        else:
            volume_efficiency = 0
        
        # Adaptive Regime Signals
        if i >= 3:
            recent_3 = current_data.iloc[-3:]
            
            # Bidirectional Pressure
            upward_pressure = np.mean((recent_3['high'] - recent_3['close']) / recent_3['close'])
            downward_pressure = np.mean((recent_3['close'] - recent_3['low']) / recent_3['close'])
            pressure_imbalance = (upward_pressure - downward_pressure) / (abs(upward_pressure) + abs(downward_pressure) + 1e-8)
            
            # Regime-Adaptive Weighting
            momentum_strength = abs(fractal_momentum)
            volume_confirmation = volume_efficiency
            
            if momentum_strength > 0.02 and volume_confirmation > 0.5:
                # Trending regime
                regime_weight = 0.7 * fractal_momentum + 0.3 * volume_efficiency
            elif abs(pressure_imbalance) < 0.3:
                # Range-bound regime
                regime_weight = 0.4 * pressure_imbalance + 0.6 * fractal_dim
            else:
                # Transition regime
                regime_weight = 0.5 * fractal_momentum + 0.5 * volume_efficiency
        else:
            regime_weight = 0
        
        # Factor Synthesis
        if i >= 20:
            # High efficiency + confirming volume + fractal momentum
            if volume_efficiency > 0.6 and fractal_momentum * regime_weight > 0:
                directional_bias = fractal_momentum * volume_efficiency * regime_weight
            
            # Low efficiency + diverging volume + conflicting momentum
            elif volume_efficiency < 0.3 and fractal_momentum * regime_weight < 0:
                directional_bias = -fractal_momentum * (1 - volume_efficiency) * regime_weight
            
            # Mixed signals with volume-volatility confluence
            else:
                momentum_acceleration = (momentum_short - momentum_medium) + (momentum_medium - momentum_long)
                directional_bias = momentum_acceleration * volume_vol_corr * regime_weight
            
            factor.iloc[i] = directional_bias
        else:
            factor.iloc[i] = 0
    
    # Fill initial values
    factor = factor.fillna(0)
    
    return factor
