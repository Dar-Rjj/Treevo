import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum-Fractal Synchronized Alpha Factor
    """
    data = df.copy()
    
    # Helper function for Hurst exponent approximation (short-term)
    def hurst_exponent_approx(series, window=5):
        """Approximate Hurst exponent using rescaled range method"""
        hurst_values = []
        for i in range(len(series)):
            if i < window:
                hurst_values.append(0.5)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < 2:
                hurst_values.append(0.5)
                continue
                
            # Calculate mean and deviations
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_deviations = deviations.cumsum()
            
            # Range and standard deviation
            R = cumulative_deviations.max() - cumulative_deviations.min()
            S = window_data.std()
            
            if S == 0 or R == 0:
                hurst_values.append(0.5)
            else:
                hurst = np.log(R/S) / np.log(window)
                hurst_values.append(min(max(hurst, 0), 1))
                
        return pd.Series(hurst_values, index=series.index)
    
    # Helper function for fractal dimension approximation
    def fractal_dimension_approx(series, window=5):
        """Approximate fractal dimension using box counting method"""
        fd_values = []
        for i in range(len(series)):
            if i < window:
                fd_values.append(1.0)
                continue
                
            window_data = series.iloc[i-window+1:i+1]
            if len(window_data) < 2:
                fd_values.append(1.0)
                continue
                
            # Simple fractal dimension approximation
            range_val = window_data.max() - window_data.min()
            if range_val == 0:
                fd_values.append(1.0)
            else:
                # Normalize and calculate approximate fractal dimension
                normalized = (window_data - window_data.min()) / range_val
                changes = normalized.diff().abs().sum()
                if changes == 0:
                    fd_values.append(1.0)
                else:
                    fd = 2 - (np.log(changes) / np.log(window))
                    fd_values.append(min(max(fd, 1.0), 2.0))
                    
        return pd.Series(fd_values, index=series.index)
    
    # 1. Momentum-Fractal Integration
    # Intraday Momentum Components
    data['high_momentum'] = (data['high'] - data['close']) / data['close']
    data['low_momentum'] = (data['close'] - data['low']) / data['close']
    
    # Momentum Persistence and Acceleration
    data['close_ret'] = data['close'].pct_change()
    data['momentum_persistence'] = np.sign(data['close_ret']) * data['close_ret']
    data['acceleration_signal'] = data['momentum_persistence'].diff()
    
    # Short-term momentum calculations
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_5d'] = data['close'].pct_change(5)
    
    # Fractal calculations
    data['hurst_short'] = hurst_exponent_approx(data['close'], window=5)
    data['fractal_dim'] = fractal_dimension_approx(data['close'], window=5)
    data['fractal_dim_change'] = data['fractal_dim'].diff()
    
    # Fractal Momentum Enhancement
    data['hurst_momentum'] = data['momentum_3d'] * data['hurst_short']
    data['fractal_acceleration'] = data['momentum_5d'] * data['fractal_dim_change']
    
    # 2. Volume-Fractal Dynamics
    # Volume Concentration Analysis
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_trend_ratio'] = data['volume'] / data['volume_ma_5d']
    data['volume_stability'] = data['volume'].rolling(window=5, min_periods=1).std() / data['volume_ma_5d']
    
    # Volume Fractal Patterns
    data['volume_scaling'] = data['volume'] / data['volume'].shift(1)
    data['volume_scaling'] = data['volume_scaling'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Volume fractal using rolling standard deviation
    data['volume_fractal'] = np.log(data['volume'].rolling(window=5, min_periods=1).std() + 1e-8) / np.log(5)
    
    # Fractal-Volume Synchronization
    data['volume_price_fractal'] = data['volume'] * data['hurst_short']
    data['multi_scale_volume_alignment'] = data['volume_trend_ratio'] * data['fractal_dim_change']
    
    # 3. Microstructure-Informed Efficiency
    # Order Flow Pressure
    up_tick_pressure = []
    down_tick_pressure = []
    
    for i in range(len(data)):
        if i < 5:
            up_tick_pressure.append(0)
            down_tick_pressure.append(0)
            continue
            
        up_pressure = 0
        down_pressure = 0
        for j in range(5):
            idx = i - j
            if idx > 0 and data['close'].iloc[idx] > data['close'].iloc[idx-1]:
                up_pressure += data['volume'].iloc[idx]
            elif idx > 0 and data['close'].iloc[idx] < data['close'].iloc[idx-1]:
                down_pressure += data['volume'].iloc[idx]
                
        up_tick_pressure.append(up_pressure)
        down_tick_pressure.append(down_pressure)
    
    data['up_tick_pressure'] = up_tick_pressure
    data['down_tick_pressure'] = down_tick_pressure
    
    # Microstructure Pressure Enhancement
    data['vwap_approx'] = data['amount'] / (data['volume'] + 1e-8)
    data['amount_weighted_pressure'] = data['up_tick_pressure'] * data['vwap_approx']
    
    data['amount_ma_5d'] = data['amount'].rolling(window=5, min_periods=1).mean()
    data['large_trade_indicator'] = (data['amount'] > 2 * data['amount_ma_5d']).astype(float)
    data['large_trade_pressure'] = data['up_tick_pressure'] * data['large_trade_indicator']
    
    # Pressure Differential Analysis
    data['relative_buy_pressure'] = data['up_tick_pressure'] / (data['up_tick_pressure'] + data['down_tick_pressure'] + 1e-8)
    data['pressure_momentum'] = data['relative_buy_pressure'].diff()
    
    # Price Efficiency Analysis
    data['price_path_efficiency'] = (data['close'] - data['close'].shift(4)) / (
        (data['high'] - data['low']).rolling(window=5, min_periods=1).sum() + 1e-8)
    data['fractal_efficiency'] = data['price_path_efficiency'] * data['hurst_short']
    
    # Volume Distribution Pattern
    data['volume_clustering'] = (data['volume'].rolling(window=5, min_periods=1).max() / 
                                data['volume'].rolling(window=5, min_periods=1).min())
    data['volume_clustering'] = data['volume_clustering'].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Volume-Price Correlation
    data['volume_price_corr'] = data['close'].rolling(window=5, min_periods=1).corr(data['volume'])
    data['volume_price_corr'] = data['volume_price_corr'].fillna(0)
    
    # Microstructure Impact
    data['temporary_impact'] = (data['high'] - data['close']) / data['close']
    data['permanent_impact'] = (data['close'] - data['open']) / data['open']
    data['impact_ratio'] = abs(data['permanent_impact'] / (data['temporary_impact'] + 1e-8))
    data['impact_adjusted_efficiency'] = data['price_path_efficiency'] / (1 + data['impact_ratio'])
    
    # 4. Volatility-Fractal Regime Adaptation
    # Multi-Scale Volatility Analysis
    data['realized_vol_3d'] = data['close_ret'].rolling(window=3, min_periods=1).std()
    data['range_volatility'] = (data['high'] - data['low']) / data['close'].shift(1)
    
    data['realized_vol_10d'] = data['close_ret'].rolling(window=10, min_periods=1).std()
    data['volatility_persistence'] = data['realized_vol_10d'] / (data['realized_vol_3d'] + 1e-8)
    data['fractal_volatility'] = data['hurst_short'] * data['realized_vol_3d']
    
    # Regime Detection
    data['volatility_regime'] = (data['realized_vol_10d'] > 0.15).astype(float)
    data['jump_detection'] = (abs(data['close_ret']) > 2 * data['realized_vol_10d']).astype(float)
    data['regime_transition'] = data['volatility_regime'].diff().abs()
    
    # Regime-Sensitive Momentum Adjustment
    data['vol_adj_high_momentum'] = data['high_momentum'] / (data['range_volatility'] + 1e-8)
    data['vol_adj_low_momentum'] = data['low_momentum'] / (data['range_volatility'] + 1e-8)
    
    data['high_vol_fractal_momentum'] = data['momentum_3d'] * data['hurst_short'] * data['volatility_regime']
    data['transition_fractal_momentum'] = data['momentum_5d'] * data['fractal_dim_change'] * data['regime_transition']
    
    # 5. Core Factor Synthesis
    # Momentum-Fractal Synchronization
    data['triple_alignment'] = (data['vol_adj_high_momentum'] * 
                               data['volume_price_fractal'] * 
                               data['pressure_momentum'])
    
    data['bidirectional_efficiency'] = (data['vol_adj_low_momentum'] * 
                                      data['volume_trend_ratio'] * 
                                      data['relative_buy_pressure'])
    
    # Microstructure-Enhanced Factors
    data['impact_adj_momentum'] = data['momentum_3d'] / (1 + data['impact_ratio'])
    data['large_trade_sync'] = data['amount_weighted_pressure'] * data['fractal_efficiency']
    
    # Efficiency-Weighted Enhancement
    data['fractal_efficiency_weighting'] = data['triple_alignment'] * data['fractal_efficiency']
    data['volume_dist_adj'] = data['bidirectional_efficiency'] / (data['volume_clustering'] + 1e-8)
    
    # 6. Multi-Scale Adaptive Refinement
    # Volatility-Fractal Optimization
    data['high_vol_scaling'] = (data['fractal_efficiency_weighting'] + data['volume_dist_adj']) / 2 * data['fractal_volatility']
    data['low_vol_enhancement'] = (data['fractal_efficiency_weighting'] + data['volume_dist_adj']) / 2 / (data['realized_vol_3d'] + 1e-8)
    
    # Trend Phase Confirmation
    data['momentum_accel_boost'] = (data['fractal_efficiency_weighting'] + data['volume_dist_adj']) / 2 * data['acceleration_signal']
    
    data['volume_ma_long'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(5)
    data['volume_breakout_validation'] = (data['fractal_efficiency_weighting'] + data['volume_dist_adj']) / 2 * (
        data['volume'] / (data['volume_ma_long'] + 1e-8))
    
    # 7. Regime-Adaptive Final Alpha
    # High Volatility Alpha
    high_vol_alpha = 0.6 * data['high_vol_scaling'] + 0.4 * data['impact_adj_momentum']
    
    # Low Volatility Alpha  
    low_vol_alpha = 0.7 * data['low_vol_enhancement'] + 0.3 * data['large_trade_sync']
    
    # Transition Alpha
    transition_alpha = 0.5 * data['momentum_accel_boost'] + 0.5 * data['volume_breakout_validation']
    
    # Final adaptive alpha based on volatility regime
    final_alpha = (
        data['volatility_regime'] * high_vol_alpha +
        (1 - data['volatility_regime']) * low_vol_alpha +
        data['regime_transition'] * transition_alpha
    ) / (1 + data['volatility_regime'] + data['regime_transition'])
    
    # Clean and normalize
    final_alpha = final_alpha.replace([np.inf, -np.inf], 0).fillna(0)
    
    return final_alpha
