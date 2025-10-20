import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Multi-Fractal Momentum-Volume Alpha Factor
    Combines fractal analysis, momentum patterns, and micro-structure dynamics
    """
    data = df.copy()
    
    # Core price and volume calculations
    data['returns'] = data['close'].pct_change()
    data['high_low_range'] = (data['high'] - data['low']) / data['close']
    data['volume_change'] = data['volume'].pct_change()
    
    # Multi-scale momentum persistence
    def hurst_exponent(series, window=20):
        """Calculate Hurst exponent using R/S analysis"""
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = series.iloc[i-window:i].dropna()
            if len(window_data) < 10:
                hurst_values.append(np.nan)
                continue
                
            # R/S analysis
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            Z = deviations.cumsum()
            R = Z.max() - Z.min()
            S = window_data.std()
            
            if S > 0:
                hurst = np.log(R/S) / np.log(window)
                hurst_values.append(hurst)
            else:
                hurst_values.append(np.nan)
        
        return pd.Series(hurst_values, index=series.index)
    
    # Calculate multi-scale Hurst exponents
    data['hurst_5'] = hurst_exponent(data['returns'], 5)
    data['hurst_10'] = hurst_exponent(data['returns'], 10)
    data['hurst_20'] = hurst_exponent(data['returns'], 20)
    
    # Volume scaling analysis
    def volume_scaling_exponent(volume_series, price_series, window=10):
        """Calculate volume scaling exponent"""
        exponents = []
        for i in range(len(volume_series)):
            if i < window:
                exponents.append(np.nan)
                continue
                
            vol_data = volume_series.iloc[i-window:i]
            price_data = price_series.iloc[i-window:i]
            
            if len(vol_data.dropna()) < 5:
                exponents.append(np.nan)
                continue
                
            # Log-log regression of volume vs price range
            log_vol = np.log(vol_data.replace(0, np.nan).dropna())
            log_range = np.log((price_data['high'] - price_data['low']).replace(0, np.nan).dropna())
            
            try:
                slope, _, _, _, _ = linregress(log_range, log_vol)
                exponents.append(slope)
            except:
                exponents.append(np.nan)
        
        return pd.Series(exponents, index=volume_series.index)
    
    data['volume_scaling'] = volume_scaling_exponent(data['volume'], data[['high', 'low']], 10)
    
    # Momentum fracture detection
    def momentum_fracture_detection(returns, window=5):
        """Detect momentum breakdown points"""
        fractures = []
        for i in range(len(returns)):
            if i < window:
                fractures.append(0)
                continue
                
            window_returns = returns.iloc[i-window:i].dropna()
            if len(window_returns) < 3:
                fractures.append(0)
                continue
                
            # Check for momentum reversal patterns
            recent_momentum = window_returns.iloc[-3:].mean()
            previous_momentum = window_returns.iloc[:-3].mean()
            
            # Fracture when strong positive momentum breaks
            if previous_momentum > 0.02 and recent_momentum < -0.01:
                fractures.append(1)
            # Fracture when strong negative momentum breaks
            elif previous_momentum < -0.02 and recent_momentum > 0.01:
                fractures.append(-1)
            else:
                fractures.append(0)
        
        return pd.Series(fractures, index=returns.index)
    
    data['momentum_fracture'] = momentum_fracture_detection(data['returns'])
    
    # Micro-structure pressure accumulation
    def microstructure_pressure(open_p, high, low, close, volume, window=3):
        """Calculate micro-structure pressure from OHLC patterns"""
        pressure = []
        for i in range(len(open_p)):
            if i < window:
                pressure.append(0)
                continue
                
            # Opening gap pressure
            gap = (open_p.iloc[i] - close.iloc[i-1]) / close.iloc[i-1]
            
            # Intraday pressure from range and close position
            daily_range = high.iloc[i] - low.iloc[i]
            if daily_range > 0:
                close_position = (close.iloc[i] - low.iloc[i]) / daily_range
            else:
                close_position = 0.5
                
            # Volume-weighted pressure
            vol_weight = volume.iloc[i] / volume.iloc[i-window:i].mean() if volume.iloc[i-window:i].mean() > 0 else 1
            
            # Combined pressure score
            pressure_score = gap * 2 + (close_position - 0.5) * 3 + np.log(vol_weight)
            pressure.append(pressure_score)
        
        return pd.Series(pressure, index=open_p.index)
    
    data['micro_pressure'] = microstructure_pressure(data['open'], data['high'], data['low'], data['close'], data['volume'])
    
    # Multi-scale regime synchronization
    def regime_synchronization(hurst_short, hurst_medium, hurst_long, returns):
        """Calculate multi-scale regime alignment"""
        sync_scores = []
        for i in range(len(returns)):
            if pd.isna(hurst_short.iloc[i]) or pd.isna(hurst_medium.iloc[i]) or pd.isna(hurst_long.iloc[i]):
                sync_scores.append(0)
                continue
                
            # Regime classification
            short_regime = 1 if hurst_short.iloc[i] > 0.6 else (-1 if hurst_short.iloc[i] < 0.4 else 0)
            medium_regime = 1 if hurst_medium.iloc[i] > 0.6 else (-1 if hurst_medium.iloc[i] < 0.4 else 0)
            long_regime = 1 if hurst_long.iloc[i] > 0.6 else (-1 if hurst_long.iloc[i] < 0.4 else 0)
            
            # Synchronization score
            if short_regime == medium_regime == long_regime:
                sync_score = 2.0 * short_regime  # Strong alignment
            elif (short_regime == medium_regime) or (medium_regime == long_regime):
                sync_score = 1.0 * ((short_regime + medium_regime + long_regime) / 3)  # Partial alignment
            else:
                sync_score = 0  # No alignment
            
            sync_scores.append(sync_score)
        
        return pd.Series(sync_scores, index=returns.index)
    
    data['regime_sync'] = regime_synchronization(data['hurst_5'], data['hurst_10'], data['hurst_20'], data['returns'])
    
    # Volume memory effects
    def volume_memory(volume, lags=[1, 3, 5]):
        """Calculate volume autocorrelation patterns"""
        memory_scores = []
        for i in range(len(volume)):
            if i < max(lags):
                memory_scores.append(0)
                continue
                
            current_vol = volume.iloc[i]
            lag_correlations = []
            
            for lag in lags:
                if i >= lag:
                    lag_vol = volume.iloc[i-lag]
                    if current_vol > 0 and lag_vol > 0:
                        correlation = min(current_vol, lag_vol) / max(current_vol, lag_vol)
                        lag_correlations.append(correlation)
            
            if lag_correlations:
                memory_score = np.mean(lag_correlations)
            else:
                memory_score = 0
                
            memory_scores.append(memory_score)
        
        return pd.Series(memory_scores, index=volume.index)
    
    data['volume_memory'] = volume_memory(data['volume'])
    
    # Composite fractal alpha construction
    def composite_fractal_alpha(data):
        """Combine all components into final alpha factor"""
        alpha_values = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            if any(pd.isna([row['hurst_5'], row['hurst_10'], row['volume_scaling'], row['regime_sync']])):
                alpha_values.append(0)
                continue
            
            # Core fractal momentum component
            fractal_momentum = (row['hurst_5'] + row['hurst_10']) / 2
            
            # Volume efficiency component
            volume_efficiency = row['volume_scaling'] if not pd.isna(row['volume_scaling']) else 0
            
            # Regime alignment component
            regime_component = row['regime_sync']
            
            # Micro-structure pressure component (smoothed)
            micro_component = row['micro_pressure'] if not pd.isna(row['micro_pressure']) else 0
            
            # Volume memory component
            volume_mem_component = row['volume_memory'] if not pd.isna(row['volume_memory']) else 0
            
            # Momentum fracture adjustment
            fracture_adj = -0.5 * row['momentum_fracture'] if not pd.isna(row['momentum_fracture']) else 0
            
            # Composite alpha with adaptive weights
            alpha = (
                0.4 * fractal_momentum +
                0.25 * volume_efficiency +
                0.2 * regime_component +
                0.1 * micro_component +
                0.05 * volume_mem_component +
                fracture_adj
            )
            
            alpha_values.append(alpha)
        
        return pd.Series(alpha_values, index=data.index)
    
    # Generate final alpha factor
    alpha_series = composite_fractal_alpha(data)
    
    # Smooth the alpha with a 3-day rolling mean (using only past data)
    alpha_smoothed = alpha_series.rolling(window=3, min_periods=1).mean()
    
    return alpha_smoothed
