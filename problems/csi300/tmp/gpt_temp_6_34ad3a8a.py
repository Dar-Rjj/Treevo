import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Momentum Acceleration Framework
    # Short-Term Momentum (3-day)
    data['return_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_change_3d'] = data['return_3d'] - data['return_3d'].shift(1)
    
    # Medium-Term Momentum (10-day)
    data['return_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['momentum_stability_10d'] = abs(data['return_10d'] - data['return_10d'].shift(1))
    
    # Acceleration Signal
    data['momentum_gap'] = data['return_3d'] - data['return_10d']
    data['acceleration_strength'] = data['momentum_gap'] - data['momentum_gap'].shift(1)
    
    # Acceleration Persistence
    def count_consecutive_sign(series):
        signs = np.sign(series)
        counts = []
        current_count = 0
        current_sign = 0
        
        for sign in signs:
            if np.isnan(sign):
                counts.append(np.nan)
                current_count = 0
                current_sign = 0
            elif sign == current_sign:
                current_count += 1
                counts.append(current_count)
            else:
                current_count = 1
                current_sign = sign
                counts.append(current_count)
        return pd.Series(counts, index=series.index)
    
    data['acceleration_persistence'] = count_consecutive_sign(data['acceleration_strength'])
    
    # Volume Confirmation Analysis
    # Volume Momentum
    data['volume_change'] = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    data['volume_acceleration'] = data['volume_change'] - data['volume_change'].shift(1)
    data['volume_trend'] = count_consecutive_sign(data['volume_change'])
    
    # Volume-Price Alignment
    data['direction_agreement'] = np.sign(data['volume_change']) * np.sign(data['return_3d'])
    data['strength_correlation'] = abs(data['volume_change']) * abs(data['return_3d'])
    data['persistence_match'] = data['volume_trend'] * data['acceleration_persistence']
    
    # Volume Stability
    data['volume_range'] = (data['volume'].rolling(window=5).max() - data['volume'].rolling(window=5).min()) / data['volume'].rolling(window=5).mean()
    data['volume_consistency'] = 1 / (1 + data['volume_range'])
    
    # Regime Adaptation
    # Volatility Context
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_range'] = data['daily_range'].rolling(window=5).mean()
    data['volatility_ratio'] = data['daily_range'] / data['avg_range']
    
    # Trend Context
    data['price_slope'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    def count_slope_consistency(series, window=5):
        consistency = []
        for i in range(len(series)):
            if i < window:
                consistency.append(np.nan)
            else:
                current_sign = np.sign(series.iloc[i])
                window_data = series.iloc[i-window+1:i+1]
                count = sum(np.sign(window_data) == current_sign)
                consistency.append(count)
        return pd.Series(consistency, index=series.index)
    
    data['slope_consistency'] = count_slope_consistency(data['return_3d'])
    data['trend_strength'] = abs(data['price_slope']) * data['slope_consistency']
    
    # Adaptive Weights
    data['volatility_adjustment'] = 1 / (1 + data['volatility_ratio'])
    data['trend_confidence'] = 1 + (data['trend_strength'] / 10)
    data['regime_multiplier'] = data['volatility_adjustment'] * data['trend_confidence']
    
    # Signal Construction
    # Core Acceleration Signal
    data['base_acceleration'] = data['acceleration_strength'] * data['direction_agreement']
    data['volume_confirmed'] = data['base_acceleration'] * data['strength_correlation']
    data['persistence_enhanced'] = data['volume_confirmed'] * data['persistence_match']
    
    # Bounded Transformation
    data['smoothed_signal'] = data['persistence_enhanced'].rolling(window=3).mean()
    data['range_limited'] = data['smoothed_signal'] / (1 + abs(data['smoothed_signal']))
    
    # Apply consistency filter
    data['bounded_signal'] = np.where(data['slope_consistency'] > 2, data['range_limited'], 0)
    
    # Multi-Timeframe Integration
    data['medium_term_confirmation'] = np.sign(data['return_10d']) * data['bounded_signal']
    data['integrated_signal'] = data['bounded_signal'] * data['medium_term_confirmation']
    
    # Final Alpha Output
    # Regime Adaptation
    data['adapted_signal'] = data['integrated_signal'] * data['regime_multiplier']
    data['volume_weighted_signal'] = data['adapted_signal'] * data['volume_consistency']
    
    # Persistence Validation
    def count_signal_consistency(series, window=3):
        consistency = []
        for i in range(len(series)):
            if i < window:
                consistency.append(np.nan)
            else:
                current_sign = np.sign(series.iloc[i])
                window_data = series.iloc[i-window+1:i+1]
                count = sum(np.sign(window_data) == current_sign)
                consistency.append(count)
        return pd.Series(consistency, index=series.index)
    
    data['signal_consistency'] = count_signal_consistency(data['volume_weighted_signal'])
    data['final_alpha'] = np.where(data['signal_consistency'] >= 2, 
                                 data['volume_weighted_signal'] * 1.5, 
                                 data['volume_weighted_signal'])
    
    # Return the final alpha factor
    return data['final_alpha']
