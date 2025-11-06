import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Helper function to calculate fractal dimension (Hurst exponent approximation)
    def hurst_exponent(series, window=20):
        if len(series) < window:
            return pd.Series(np.nan, index=series.index)
        
        hurst_values = []
        for i in range(len(series)):
            if i < window - 1:
                hurst_values.append(np.nan)
                continue
            
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < window:
                hurst_values.append(np.nan)
                continue
            
            # Calculate Hurst exponent using R/S method
            lags = range(2, min(10, len(window_data)))
            tau = []
            for lag in lags:
                # Calculate rescaled range
                series_lag = []
                for j in range(0, len(window_data), lag):
                    if j + lag <= len(window_data):
                        sub_series = window_data.iloc[j:j+lag]
                        mean_val = sub_series.mean()
                        cumulative_deviation = (sub_series - mean_val).cumsum()
                        R = cumulative_deviation.max() - cumulative_deviation.min()
                        S = sub_series.std()
                        if S > 0:
                            series_lag.append(R / S)
                
                if series_lag:
                    tau.append(np.log(np.mean(series_lag)))
            
            if len(tau) > 1:
                x = np.log(lags[:len(tau)])
                y = np.array(tau)
                if len(x) > 1 and not np.any(np.isnan(y)):
                    hurst = np.polyfit(x, y, 1)[0]
                    hurst_values.append(hurst)
                else:
                    hurst_values.append(np.nan)
            else:
                hurst_values.append(np.nan)
        
        return pd.Series(hurst_values, index=series.index)
    
    # Calculate fractal dimensions
    price_fractal_dimension = hurst_exponent(data['close'], window=20)
    volume_fractal_dimension = hurst_exponent(data['volume'], window=20)
    
    # Fractal Volatility Dynamics
    fractal_true_range = (data['high'] - data['low']) / np.log(data['volume'] + 1e-8)
    
    gap_fractal_volatility = np.abs(data['open'] - data['close'].shift(1)) / np.log(data['high'] - data['low'] + 1e-8)
    gap_fractal_volatility = gap_fractal_volatility.replace([np.inf, -np.inf], np.nan)
    
    fractal_volatility_breakout = (data['high'] - data['low']) > (1.5 * (data['high'].shift(1) - data['low'].shift(1)) * price_fractal_dimension)
    fractal_volatility_breakout = fractal_volatility_breakout.astype(float)
    
    # Fractal-Volume Coherence
    fractal_volume_acceleration = (data['volume'] / data['volume'].shift(1)) * volume_fractal_dimension
    
    fractal_volume_price_alignment = data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-6) * price_fractal_dimension
    fractal_volume_price_alignment = fractal_volume_price_alignment.replace([np.inf, -np.inf], np.nan)
    
    fractal_volume_pressure = data['amount'] / (data['high'] - data['low'] + 1e-6) * volume_fractal_dimension
    fractal_volume_pressure = fractal_volume_pressure.replace([np.inf, -np.inf], np.nan)
    
    # Fractal Efficiency Analysis
    fractal_opening_efficiency = ((data['open'] / data['close'].shift(1) - 1) / np.log(data['high'].shift(1) - data['low'].shift(1) + 1e-6))
    fractal_opening_efficiency = fractal_opening_efficiency.replace([np.inf, -np.inf], np.nan)
    
    fractal_normalized_change = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-6)) * price_fractal_dimension
    fractal_normalized_change = fractal_normalized_change.replace([np.inf, -np.inf], np.nan)
    
    fractal_price_impact = np.abs(data['close'] - data['open']) / np.log(data['high'] - data['low'] + 1e-8)
    fractal_price_impact = fractal_price_impact.replace([np.inf, -np.inf], np.nan)
    
    # Adaptive Fractal Alpha Synthesis
    fractal_regime_component = fractal_volatility_breakout * fractal_volume_acceleration
    
    fractal_coherence_component = fractal_volume_price_alignment * fractal_volume_pressure
    
    fractal_efficiency_component = fractal_opening_efficiency * fractal_normalized_change * fractal_price_impact
    
    # Final Alpha
    final_alpha = fractal_regime_component * fractal_coherence_component * fractal_efficiency_component
    
    # Clean and normalize
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    final_alpha = (final_alpha - final_alpha.rolling(window=20, min_periods=10).mean()) / final_alpha.rolling(window=20, min_periods=10).std()
    
    return final_alpha
