import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Volume Momentum with Dynamic Pattern Recognition
    """
    data = df.copy()
    
    # Initialize all required columns to avoid KeyError
    data['volatility_5d'] = 0.0
    data['volatility_20d'] = 0.0
    data['volatility_ratio'] = 0.0
    data['volatility_persistence'] = 0
    data['intraday_range_efficiency'] = 0.0
    data['volume_weighted_efficiency'] = 0.0
    data['volume_distribution_asymmetry'] = 0.0
    data['volume_acceleration'] = 0.0
    data['volume_acceleration_persistence'] = 0
    data['volatility_regime_shift'] = 0.0
    data['volume_volatility_sync'] = 0.0
    data['momentum_5d'] = 0.0
    data['momentum_20d'] = 0.0
    data['volume_momentum'] = 0.0
    data['momentum_divergence'] = 0.0
    data['price_rejection'] = 0.0
    data['volume_spike_asymmetry'] = 0.0
    data['volume_distribution_efficiency'] = 0.0
    data['pattern_persistence'] = 0
    
    # Multi-Timeframe Volatility Analysis
    for i in range(len(data)):
        if i >= 4:  # 5-day volatility
            high_low_range_5d = data['high'].iloc[i-4:i+1].max() - data['low'].iloc[i-4:i+1].min()
            data.loc[data.index[i], 'volatility_5d'] = high_low_range_5d / data['close'].iloc[i-4:i+1].mean()
        
        if i >= 19:  # 20-day volatility
            high_low_range_20d = data['high'].iloc[i-19:i+1].max() - data['low'].iloc[i-19:i+1].min()
            data.loc[data.index[i], 'volatility_20d'] = high_low_range_20d / data['close'].iloc[i-19:i+1].mean()
            
            # Volatility Ratio
            if data['volatility_20d'].iloc[i] != 0:
                data.loc[data.index[i], 'volatility_ratio'] = data['volatility_5d'].iloc[i] / data['volatility_20d'].iloc[i]
    
    # Volatility Persistence Scoring
    for i in range(1, len(data)):
        if i >= 1:
            vol_dir_current = 1 if data['volatility_5d'].iloc[i] > data['volatility_5d'].iloc[i-1] else -1
            if i >= 2:
                vol_dir_prev = 1 if data['volatility_5d'].iloc[i-1] > data['volatility_5d'].iloc[i-2] else -1
                if vol_dir_current == vol_dir_prev:
                    data.loc[data.index[i], 'volatility_persistence'] = data['volatility_persistence'].iloc[i-1] + 1
                else:
                    data.loc[data.index[i], 'volatility_persistence'] = 1
    
    # Volume-Enhanced Price Fractal Analysis
    for i in range(1, len(data)):
        # Intraday Range Efficiency
        if i >= 1:
            price_change = abs(data['close'].iloc[i] - data['close'].iloc[i-1])
            if price_change != 0:
                data.loc[data.index[i], 'intraday_range_efficiency'] = (
                    (data['high'].iloc[i] - data['low'].iloc[i]) / price_change
                )
        
        # Volume-Weighted Price Path Efficiency
        high_low_range = data['high'].iloc[i] - data['low'].iloc[i]
        if high_low_range != 0:
            price_efficiency = (data['close'].iloc[i] - data['open'].iloc[i]) / high_low_range
            # Volume intensity factor (current volume vs 5-day avg)
            if i >= 4:
                volume_5d_avg = data['volume'].iloc[i-4:i+1].mean()
                volume_intensity = data['volume'].iloc[i] / volume_5d_avg if volume_5d_avg != 0 else 1.0
                data.loc[data.index[i], 'volume_weighted_efficiency'] = price_efficiency * volume_intensity
    
    # Multi-Scale Volume Momentum Analysis
    for i in range(len(data)):
        # Volume Distribution Asymmetry
        if i >= 4:
            volume_1d = data['volume'].iloc[i]
            volume_5d_avg = data['volume'].iloc[i-4:i+1].mean()
            if volume_5d_avg != 0:
                data.loc[data.index[i], 'volume_distribution_asymmetry'] = volume_1d / volume_5d_avg
        
        # Volume Acceleration Patterns
        if i >= 5:
            volume_change_1d = data['volume'].iloc[i] - data['volume'].iloc[i-1]
            volume_change_5d = data['volume'].iloc[i] - data['volume'].iloc[i-5]
            if volume_change_5d != 0:
                data.loc[data.index[i], 'volume_acceleration'] = volume_change_1d / abs(volume_change_5d)
    
    # Volume Acceleration Persistence
    for i in range(1, len(data)):
        if i >= 1:
            accel_dir_current = 1 if data['volume_acceleration'].iloc[i] > data['volume_acceleration'].iloc[i-1] else -1
            if i >= 2:
                accel_dir_prev = 1 if data['volume_acceleration'].iloc[i-1] > data['volume_acceleration'].iloc[i-2] else -1
                if accel_dir_current == accel_dir_prev:
                    data.loc[data.index[i], 'volume_acceleration_persistence'] = data['volume_acceleration_persistence'].iloc[i-1] + 1
                else:
                    data.loc[data.index[i], 'volume_acceleration_persistence'] = 1
    
    # Dynamic Pattern Recognition with Momentum Divergence
    for i in range(len(data)):
        # Volatility Regime Shift Detection
        if i >= 19:
            current_vol = data['volatility_5d'].iloc[i]
            vol_20d_median = data['volatility_5d'].iloc[i-19:i+1].median()
            vol_20d_std = data['volatility_5d'].iloc[i-19:i+1].std()
            if vol_20d_std != 0:
                data.loc[data.index[i], 'volatility_regime_shift'] = (current_vol - vol_20d_median) / vol_20d_std
        
        # Volume-Volatility Synchronization
        if i >= 4:
            vol_change = data['volatility_5d'].iloc[i] - data['volatility_5d'].iloc[i-1]
            volume_change = data['volume'].iloc[i] - data['volume'].iloc[i-1]
            if vol_change != 0:
                data.loc[data.index[i], 'volume_volatility_sync'] = volume_change / abs(vol_change)
    
    # Multi-Timeframe Momentum Divergence
    for i in range(len(data)):
        # Price Momentum Analysis
        if i >= 5:
            data.loc[data.index[i], 'momentum_5d'] = data['close'].iloc[i] / data['close'].iloc[i-5] - 1
        if i >= 20:
            data.loc[data.index[i], 'momentum_20d'] = data['close'].iloc[i] / data['close'].iloc[i-20] - 1
            
            # Momentum Divergence
            data.loc[data.index[i], 'momentum_divergence'] = data['momentum_5d'].iloc[i] - data['momentum_20d'].iloc[i]
        
        # Volume Momentum
        if i >= 5:
            if data['volume'].iloc[i-5] != 0:
                data.loc[data.index[i], 'volume_momentum'] = data['volume'].iloc[i] / data['volume'].iloc[i-5] - 1
    
    # Pattern Efficiency Analysis
    for i in range(len(data)):
        # Price Rejection
        high_low_range = data['high'].iloc[i] - data['low'].iloc[i]
        if high_low_range != 0:
            data.loc[data.index[i], 'price_rejection'] = (data['close'].iloc[i] - data['open'].iloc[i]) / high_low_range
    
    # Volume-Price Pattern Recognition
    for i in range(len(data)):
        # Volume Spike Asymmetry Analysis
        if i >= 4:
            volume_5d_median = data['volume'].iloc[i-4:i+1].median()
            if volume_5d_median != 0:
                volume_ratio = data['volume'].iloc[i] / volume_5d_median
                data.loc[data.index[i], 'volume_spike_asymmetry'] = np.log(volume_ratio) if volume_ratio > 0 else 0
        
        # Volume Distribution Efficiency
        if i >= 4:
            volume_ratios = []
            for j in range(i-4, i+1):
                if j >= 4:
                    vol_5d_median_j = data['volume'].iloc[j-4:j+1].median()
                    if vol_5d_median_j != 0:
                        volume_ratios.append(data['volume'].iloc[j] / vol_5d_median_j)
            
            if len(volume_ratios) > 0:
                max_ratio = max(volume_ratios)
                mean_ratio = np.mean(volume_ratios)
                if mean_ratio != 0:
                    data.loc[data.index[i], 'volume_distribution_efficiency'] = max_ratio / mean_ratio
    
    # Pattern Persistence Scoring
    for i in range(1, len(data)):
        if i >= 1:
            rejection_current = 1 if abs(data['price_rejection'].iloc[i]) > 0.5 else 0
            if i >= 2:
                rejection_prev = 1 if abs(data['price_rejection'].iloc[i-1]) > 0.5 else 0
                if rejection_current == rejection_prev == 1:
                    data.loc[data.index[i], 'pattern_persistence'] = data['pattern_persistence'].iloc[i-1] + 1
                else:
                    data.loc[data.index[i], 'pattern_persistence'] = rejection_current
    
    # Composite Alpha Generation
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 20:  # Ensure sufficient history
            # Fractal Volatility-Volume Component
            fractal_component = (
                data['volatility_ratio'].iloc[i] * 
                (1 + 0.1 * data['volatility_persistence'].iloc[i]) *
                data['volume_weighted_efficiency'].iloc[i] *
                (1 + 0.05 * data['volume_acceleration_persistence'].iloc[i])
            )
            
            # Dynamic Pattern Recognition Component
            pattern_component = (
                data['momentum_divergence'].iloc[i] *
                (1 + 0.2 * abs(data['volatility_regime_shift'].iloc[i])) *
                data['price_rejection'].iloc[i] *
                (1 + 0.1 * data['pattern_persistence'].iloc[i])
            )
            
            # Efficiency Analysis Component
            efficiency_component = (
                data['intraday_range_efficiency'].iloc[i] *
                data['volume_distribution_efficiency'].iloc[i] *
                (1 + 0.15 * data['volume_spike_asymmetry'].iloc[i])
            )
            
            # Volatility-Regime Based Filtering
            volatility_regime = 'high' if data['volatility_ratio'].iloc[i] > 1.2 else 'low'
            
            if volatility_regime == 'high':
                # Emphasize trend-following in high volatility
                composite_score = (
                    0.4 * fractal_component +
                    0.5 * pattern_component +
                    0.1 * efficiency_component
                )
            else:
                # Emphasize mean-reversion in low volatility
                composite_score = (
                    0.3 * fractal_component +
                    0.3 * pattern_component +
                    0.4 * efficiency_component
                )
            
            # Volume-Confirmed Enhancement
            volume_confirmation = 1 + 0.2 * data['volume_volatility_sync'].iloc[i]
            alpha_signal.iloc[i] = composite_score * volume_confirmation
    
    # Fill NaN values with 0
    alpha_signal = alpha_signal.fillna(0)
    
    return alpha_signal
