import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Helper functions for fractal dimension and chaotic measures
    def fractal_dimension(series, window=5):
        """Calculate approximate fractal dimension using Hurst exponent method"""
        if len(series) < window:
            return pd.Series(np.ones(len(series)), index=series.index)
        
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(1.0)
                continue
            
            window_data = series.iloc[i-window:i+1]
            if window_data.std() == 0:
                hurst_values.append(1.0)
                continue
            
            # Simplified Hurst calculation
            lags = range(2, min(6, len(window_data)))
            tau = [np.sqrt(np.std(np.subtract(window_data[lag:].values, window_data[:-lag].values))) 
                  for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst_values.append(poly[0])
        
        return pd.Series(hurst_values, index=series.index)
    
    def chaotic_measure(series, window=3):
        """Calculate chaotic measure using Lyapunov exponent approximation"""
        if len(series) < window + 1:
            return pd.Series(np.zeros(len(series)), index=series.index)
        
        chaotic_vals = []
        for i in range(len(series)):
            if i < window + 1:
                chaotic_vals.append(0.0)
                continue
            
            window_data = series.iloc[i-window:i+1]
            if window_data.std() == 0:
                chaotic_vals.append(0.0)
                continue
            
            # Simplified Lyapunov approximation
            diffs = np.diff(window_data.values)
            if len(diffs) < 2:
                chaotic_vals.append(0.0)
                continue
            
            lyap = np.mean(np.log(np.abs(diffs[1:] / diffs[:-1])))
            chaotic_vals.append(lyap)
        
        return pd.Series(chaotic_vals, index=series.index)
    
    def price_attractor(close, window=5):
        """Calculate price attractor using moving average convergence"""
        if len(close) < window:
            return pd.Series(np.ones(len(close)), index=close.index)
        
        ma_short = close.rolling(window=3, min_periods=1).mean()
        ma_long = close.rolling(window=window, min_periods=1).mean()
        attractor = (ma_short - ma_long) / (close.rolling(window=window, min_periods=1).std() + 1e-8)
        return attractor
    
    def critical_transition(volume, close, window=5):
        """Calculate critical transition probability"""
        if len(volume) < window:
            return pd.Series(np.zeros(len(volume)), index=volume.index)
        
        volume_ma = volume.rolling(window=window, min_periods=1).mean()
        volume_std = volume.rolling(window=window, min_periods=1).std()
        price_change = close.pct_change().abs()
        
        critical = (volume - volume_ma) / (volume_std + 1e-8) * price_change
        return critical
    
    # Calculate base components
    df['fractal_dim'] = fractal_dimension(df['close'])
    df['chaotic_meas'] = chaotic_measure(df['close'])
    df['price_attr'] = price_attractor(df['close'])
    df['critical_trans'] = critical_transition(df['volume'], df['close'])
    
    # Quantum-Fractal Volatility Components
    df['qf_upside_eff'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) * 
                          np.abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8) * 
                          df['volume'] / (df['amount'] + 1e-8) * df['fractal_dim'])
    
    df['qf_downside_eff'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * 
                            np.abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8) * 
                            df['volume'] / (df['amount'] + 1e-8) * df['fractal_dim'])
    
    df['qf_vol_ratio'] = ((df['high'] - df['open']) / (df['open'] - df['low'] + 1e-8) * 
                         df['volume'] / (df['volume'].shift(1) + 1e-8) * df['chaotic_meas'])
    
    # Quantum-Fractal Volume Dynamics
    df['qf_buy_conc'] = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * 
                        df['volume'] / (df['amount'] + 1e-8) * df['price_attr'])
    
    df['qf_sell_conc'] = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8) * 
                         df['volume'] / (df['amount'] + 1e-8) * df['price_attr'])
    
    df['qf_volume_flow'] = (((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) - 
                           (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)) * 
                           df['volume'] / (df['volume'].shift(1) + 1e-8) * df['critical_trans'])
    
    # Quantum-Fractal Divergence Patterns
    df['qf_pos_momentum'] = ((df['close'] - df['close'].shift(1)) / 
                           (np.abs(df['close'].shift(1) - df['close'].shift(2)) + 1e-8) * 
                           df['volume'] / (df['volume'].shift(1) + 1e-8) - 
                           (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) * 
                           df['fractal_dim'])
    
    df['qf_neg_momentum'] = ((df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8) - 
                           (df['close'] - df['close'].shift(1)) / 
                           (np.abs(df['close'].shift(1) - df['close'].shift(2)) + 1e-8) * 
                           df['volume'] / (df['volume'].shift(1) + 1e-8) * df['fractal_dim'])
    
    df['qf_consistency'] = (np.sign(df['close'] - df['close'].shift(1)) * 
                          np.sign(df['volume'] / (df['volume'].shift(1) + 1e-8) - 
                                 (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)) * 
                          df['chaotic_meas'])
    
    # Quantum-Fractal Efficiency Metrics
    df['chaotic_efficiency'] = (df['volume'] / (np.abs(df['close'] - df['close'].shift(1)) + 1e-8) * 
                              df['fractal_dim'] * df['qf_consistency'])
    
    df['regime_efficiency'] = (np.abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) * 
                             df['volume'] / (df['volume'].shift(1) + 1e-8) * df['qf_volume_flow'])
    
    df['critical_efficiency'] = (df['critical_trans'] * df['price_attr'] * df['qf_vol_ratio'])
    
    # Quantum-Fractal Transition Dynamics
    df['asymmetry_convergence'] = ((df['qf_upside_eff'] - df['qf_downside_eff']) + 
                                 (df['qf_buy_conc'] - df['qf_sell_conc']) + 
                                 (df['qf_pos_momentum'] - df['qf_neg_momentum']))
    
    df['asymmetry_acceleration'] = (df['qf_volume_flow'] * 
                                  (df['volume'] / (df['volume'].shift(1) + 1e-8) - 
                                   df['volume'].shift(1) / (df['volume'].shift(2) + 1e-8)) * 
                                  np.abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8) * 
                                  df['critical_trans'])
    
    # Hierarchical Quantum-Fractal Alpha Assembly
    volatility_component = (df['qf_upside_eff'] * 0.3 + 
                          df['qf_downside_eff'] * 0.3 + 
                          df['qf_vol_ratio'] * 0.4)
    
    volume_component = (df['qf_buy_conc'] * 0.4 + 
                      df['qf_sell_conc'] * 0.4 + 
                      df['qf_volume_flow'] * 0.2)
    
    divergence_component = (df['qf_pos_momentum'] * 0.4 + 
                          df['qf_neg_momentum'] * 0.35 + 
                          df['qf_consistency'] * 0.25)
    
    efficiency_component = (df['chaotic_efficiency'] * 0.4 + 
                          df['regime_efficiency'] * 0.35 + 
                          df['critical_efficiency'] * 0.25)
    
    context_multiplier = volatility_component * volume_component * 0.6
    
    # Regime classification and multiplier
    regime_multiplier = 1.0
    high_asymmetry = (df['qf_vol_ratio'] > 1.2) & (df['qf_volume_flow'] > 0.3) & (df['qf_pos_momentum'] > 0.5)
    medium_asymmetry = (df['qf_vol_ratio'] > 0.8) & (df['qf_volume_flow'] > 0.1) & (df['qf_consistency'] > 0)
    low_asymmetry = (df['qf_vol_ratio'] < 0.6) & (df['qf_volume_flow'] < -0.2) & (df['qf_neg_momentum'] > 0.3)
    
    regime_multiplier = np.where(high_asymmetry, 1.4, 
                                np.where(medium_asymmetry, 1.1, 
                                        np.where(low_asymmetry, 0.7, 1.0)))
    
    # Transition adjustment
    transition_adjustment = np.where(df['asymmetry_acceleration'] > 0.5, 0.2, 
                                   np.where(df['asymmetry_convergence'] < -0.5, -0.15, 0.0))
    
    # Final quantum-fractal alpha
    quantum_fractal_alpha = ((context_multiplier * divergence_component * efficiency_component) * 
                           regime_multiplier + transition_adjustment)
    
    return quantum_fractal_alpha
