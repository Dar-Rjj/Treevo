import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Fractal Dynamics with Momentum-Volatility Coupling
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Required rolling windows for multi-scale analysis
    windows = [5, 10, 20]
    
    for i in range(max(windows), len(data)):
        current_data = data.iloc[:i+1]
        
        # 1. Analyze Fractal Price Patterns
        price_fractal_dims = []
        price_efficiencies = []
        
        for window in windows:
            if i >= window:
                window_data = current_data.iloc[i-window+1:i+1]
                
                # Compute multi-scale price fractal dimension using high-low range
                high_low_range = (window_data['high'] - window_data['low']).mean()
                close_range = window_data['close'].max() - window_data['close'].min()
                
                if high_low_range > 0 and close_range > 0:
                    fractal_dim = np.log(window) / np.log(window / (close_range / high_low_range))
                    price_fractal_dims.append(fractal_dim)
                
                # Calculate price path efficiency
                price_changes = window_data['close'].diff().abs().sum()
                net_move = abs(window_data['close'].iloc[-1] - window_data['close'].iloc[0])
                
                if price_changes > 0:
                    efficiency = net_move / price_changes
                    price_efficiencies.append(efficiency)
        
        # 2. Quantify Volume-Price Fractal Coupling
        volume_price_coupling = []
        
        for window in windows:
            if i >= window:
                window_data = current_data.iloc[i-window+1:i+1]
                
                # Volume-weighted price fractal dimension
                volume_weighted_high_low = (window_data['high'] - window_data['low']) * window_data['volume']
                avg_weighted_range = volume_weighted_high_low.mean()
                close_range = window_data['close'].max() - window_data['close'].min()
                
                if avg_weighted_range > 0 and close_range > 0:
                    vol_weighted_fractal = np.log(window) / np.log(window / (close_range / (avg_weighted_range / window_data['volume'].mean())))
                    
                    # Volume fractal dimension using volume clustering
                    volume_changes = window_data['volume'].pct_change().abs().sum()
                    if volume_changes > 0:
                        volume_fractal = np.log(window) / np.log(window / volume_changes)
                        
                        # Measure volume-price fractal coupling strength
                        coupling_strength = 1 - abs(vol_weighted_fractal - volume_fractal) / max(vol_weighted_fractal, volume_fractal)
                        volume_price_coupling.append(coupling_strength)
        
        # 3. Analyze Momentum-Volatility Fractal Dynamics
        momentum_vol_correlations = []
        
        for window in windows:
            if i >= window:
                window_data = current_data.iloc[i-window+1:i+1]
                
                # Momentum fractal dimension
                returns = window_data['close'].pct_change().dropna()
                if len(returns) > 1:
                    momentum_changes = returns.diff().abs().sum()
                    if momentum_changes > 0:
                        momentum_fractal = np.log(len(returns)) / np.log(len(returns) / momentum_changes)
                    
                    # Volatility fractal dimension
                    daily_ranges = (window_data['high'] - window_data['low']) / window_data['close'].shift(1)
                    vol_changes = daily_ranges.diff().abs().sum()
                    
                    if vol_changes > 0 and len(returns) > 1:
                        volatility_fractal = np.log(len(returns)) / np.log(len(returns) / vol_changes)
                        
                        # Quantify momentum-volatility fractal correlation
                        if not np.isnan(momentum_fractal) and not np.isnan(volatility_fractal):
                            mom_vol_corr = 1 - abs(momentum_fractal - volatility_fractal) / max(momentum_fractal, volatility_fractal)
                            momentum_vol_correlations.append(mom_vol_corr)
        
        # 4. Assess Multi-Scale Fractal Persistence
        fractal_persistence = []
        
        for window in windows:
            if i >= window * 2:
                # Compare fractal dimensions across consecutive windows
                recent_window = current_data.iloc[i-window+1:i+1]
                previous_window = current_data.iloc[i-2*window+1:i-window+1]
                
                # Calculate fractal dimension for both windows
                recent_range = recent_window['high'].max() - recent_window['low'].min()
                recent_volatility = (recent_window['high'] - recent_window['low']).mean()
                
                previous_range = previous_window['high'].max() - previous_window['low'].min()
                previous_volatility = (previous_window['high'] - previous_window['low']).mean()
                
                if recent_volatility > 0 and previous_volatility > 0:
                    recent_fractal = np.log(window) / np.log(window / (recent_range / recent_volatility))
                    previous_fractal = np.log(window) / np.log(window / (previous_range / previous_volatility))
                    
                    # Measure persistence
                    persistence = 1 - abs(recent_fractal - previous_fractal) / max(recent_fractal, previous_fractal)
                    fractal_persistence.append(persistence)
        
        # 5. Incorporate Fractal-Momentum Interactions
        fractal_momentum_interactions = []
        
        for window in windows:
            if i >= window:
                window_data = current_data.iloc[i-window+1:i+1]
                
                # Momentum efficiency within fractal regimes
                total_price_move = abs(window_data['close'].iloc[-1] - window_data['close'].iloc[0])
                cumulative_volatility = (window_data['high'] - window_data['low']).sum()
                
                if cumulative_volatility > 0:
                    momentum_efficiency = total_price_move / cumulative_volatility
                    
                    # Fractal-dependent momentum scaling
                    high_low_range = (window_data['high'] - window_data['low']).mean()
                    close_range = window_data['close'].max() - window_data['close'].min()
                    
                    if high_low_range > 0 and close_range > 0:
                        fractal_dim = np.log(window) / np.log(window / (close_range / high_low_range))
                        
                        # Scale momentum by fractal dimension
                        scaled_momentum = momentum_efficiency * fractal_dim
                        fractal_momentum_interactions.append(scaled_momentum)
        
        # 6. Generate Final Alpha Factor
        alpha_components = []
        
        # Combine valid components with appropriate weights
        if price_fractal_dims:
            alpha_components.append(np.mean(price_fractal_dims))
        
        if price_efficiencies:
            alpha_components.append(np.mean(price_efficiencies))
        
        if volume_price_coupling:
            alpha_components.append(np.mean(volume_price_coupling))
        
        if momentum_vol_correlations:
            alpha_components.append(np.mean(momentum_vol_correlations))
        
        if fractal_persistence:
            alpha_components.append(np.mean(fractal_persistence))
        
        if fractal_momentum_interactions:
            alpha_components.append(np.mean(fractal_momentum_interactions))
        
        # Calculate final alpha value
        if alpha_components:
            # Apply volume-confirmed weighting
            recent_volume = current_data['volume'].iloc[max(0, i-4):i+1].mean()
            volume_weight = np.log1p(recent_volume) / np.log1p(current_data['volume'].iloc[:i+1].mean())
            
            # Regime-dependent scaling based on recent volatility
            recent_volatility = (current_data['high'].iloc[max(0, i-4):i+1] - current_data['low'].iloc[max(0, i-4):i+1]).mean()
            avg_volatility = (current_data['high'].iloc[:i+1] - current_data['low'].iloc[:i+1]).mean()
            
            if avg_volatility > 0:
                regime_scale = recent_volatility / avg_volatility
            else:
                regime_scale = 1.0
            
            final_alpha = np.mean(alpha_components) * volume_weight * regime_scale
            result.iloc[i] = final_alpha
        else:
            result.iloc[i] = 0
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
