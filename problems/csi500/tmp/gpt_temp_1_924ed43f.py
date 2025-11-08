import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Scaled Multi-Timeframe Momentum with Volume Confirmation
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Small epsilon to avoid division by zero
    epsilon = 1e-8
    
    # Multi-Timeframe Momentum Calculation
    # Short-Term Momentum (1-day)
    data['short_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'] + epsilon)
    
    # Medium-Term Momentum (3-day)
    data['high_3d'] = data['high'].rolling(window=3, min_periods=3).max()
    data['low_3d'] = data['low'].rolling(window=3, min_periods=3).min()
    data['medium_momentum'] = (data['close'] - data['close'].shift(3)) / (data['high_3d'] - data['low_3d'] + epsilon)
    
    # Long-Term Momentum (5-day)
    data['high_5d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['long_momentum'] = (data['close'] - data['close'].shift(5)) / (data['high_5d'] - data['low_5d'] + epsilon)
    
    # Momentum Persistence with Exponential Decay
    def calculate_persistence(momentum_series, window, decay_factor):
        persistence = pd.Series(0, index=momentum_series.index)
        weighted_persistence = pd.Series(0, index=momentum_series.index)
        
        for i in range(window, len(momentum_series)):
            window_data = momentum_series.iloc[i-window:i]
            current_momentum = momentum_series.iloc[i]
            
            if not np.isnan(current_momentum):
                # Count consecutive same-sign momentum
                same_sign_count = 0
                decay_weight = 1.0
                
                for j in range(window-1, -1, -1):
                    if j < len(window_data) and not np.isnan(window_data.iloc[j]):
                        if np.sign(window_data.iloc[j]) == np.sign(current_momentum):
                            same_sign_count += decay_weight
                            decay_weight *= decay_factor
                        else:
                            break
                
                persistence.iloc[i] = same_sign_count
                weighted_persistence.iloc[i] = same_sign_count * abs(current_momentum)
        
        return weighted_persistence
    
    # Calculate persistence for each timeframe
    data['short_persistence'] = calculate_persistence(data['short_momentum'], window=3, decay_factor=0.8)
    data['medium_persistence'] = calculate_persistence(data['medium_momentum'], window=5, decay_factor=0.9)
    data['long_persistence'] = calculate_persistence(data['long_momentum'], window=8, decay_factor=0.95)
    
    # Volume Confirmation Mechanism
    # Multi-Timeframe Volume Analysis
    data['short_volume_ratio'] = data['volume'] / (data['volume'].shift(1) + epsilon)
    data['medium_volume_trend'] = data['volume'] / (data['volume'].rolling(window=3, min_periods=3).mean().shift(1) + epsilon)
    data['long_volume_persistence'] = data['volume'] / (data['volume'].rolling(window=5, min_periods=5).mean().shift(1) + epsilon)
    
    # Volume-Momentum Alignment
    def volume_confirmation(momentum, short_vol, medium_vol, long_vol):
        if np.isnan(momentum):
            return 0
        
        momentum_sign = np.sign(momentum)
        vol_confirmation = 0
        
        # Short-term volume confirmation
        if short_vol > 1 and momentum_sign > 0:
            vol_confirmation += 1
        elif short_vol < 1 and momentum_sign < 0:
            vol_confirmation += 1
        
        # Medium-term volume confirmation
        if medium_vol > 1 and momentum_sign > 0:
            vol_confirmation += 1
        elif medium_vol < 1 and momentum_sign < 0:
            vol_confirmation += 1
        
        # Long-term volume confirmation
        if long_vol > 1 and momentum_sign > 0:
            vol_confirmation += 1
        elif long_vol < 1 and momentum_sign < 0:
            vol_confirmation += 1
        
        return vol_confirmation / 3.0  # Normalize to 0-1
    
    data['short_vol_confirmation'] = data.apply(
        lambda x: volume_confirmation(x['short_momentum'], x['short_volume_ratio'], 
                                    x['medium_volume_trend'], x['long_volume_persistence']), axis=1
    )
    data['medium_vol_confirmation'] = data.apply(
        lambda x: volume_confirmation(x['medium_momentum'], x['short_volume_ratio'], 
                                    x['medium_volume_trend'], x['long_volume_persistence']), axis=1
    )
    data['long_vol_confirmation'] = data.apply(
        lambda x: volume_confirmation(x['long_momentum'], x['short_volume_ratio'], 
                                    x['medium_volume_trend'], x['long_volume_persistence']), axis=1
    )
    
    # Volatility-Scaled Factor Construction
    # Multi-Timeframe Volatility Context
    data['short_volatility'] = data['high'] - data['low']
    data['medium_volatility'] = (data['high'] - data['low']).rolling(window=3, min_periods=3).mean()
    data['long_volatility'] = (data['high'] - data['low']).rolling(window=5, min_periods=5).mean()
    
    # Volatility-Adjusted Combination
    def volatility_scaled_signal(momentum, persistence, vol_confirmation, volatility):
        if np.isnan(momentum) or np.isnan(persistence) or np.isnan(vol_confirmation) or np.isnan(volatility):
            return 0
        
        if volatility == 0:
            return 0
        
        # Scale momentum by volatility, weight by persistence and volume confirmation
        scaled_momentum = momentum / (volatility + epsilon)
        weighted_signal = scaled_momentum * persistence * (1 + vol_confirmation)
        
        return weighted_signal
    
    # Calculate volatility-scaled signals for each timeframe
    data['short_scaled_signal'] = data.apply(
        lambda x: volatility_scaled_signal(x['short_momentum'], x['short_persistence'], 
                                         x['short_vol_confirmation'], x['short_volatility']), axis=1
    )
    data['medium_scaled_signal'] = data.apply(
        lambda x: volatility_scaled_signal(x['medium_momentum'], x['medium_persistence'], 
                                         x['medium_vol_confirmation'], x['medium_volatility']), axis=1
    )
    data['long_scaled_signal'] = data.apply(
        lambda x: volatility_scaled_signal(x['long_momentum'], x['long_persistence'], 
                                         x['long_vol_confirmation'], x['long_volatility']), axis=1
    )
    
    # Final Alpha Factor with inverse volatility scaling
    avg_volatility = (data['short_volatility'] + data['medium_volatility'] + data['long_volatility']) / 3
    combined_signal = (data['short_scaled_signal'] + data['medium_scaled_signal'] + data['long_scaled_signal']) / 3
    
    # Apply inverse volatility scaling to final factor
    final_factor = combined_signal / (avg_volatility + epsilon)
    
    return final_factor
