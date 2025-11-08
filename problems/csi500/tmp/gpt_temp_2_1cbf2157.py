import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate required components
    data['returns'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['high_low_range'] = data['high'] - data['low']
    data['prev_close_range'] = abs(data['high'] - data['close'].shift(1))
    data['true_range'] = data[['high_low_range', 'prev_close_range']].max(axis=1)
    
    # Calculate rolling statistics
    for i in range(len(data)):
        if i < 20:  # Need enough data for calculations
            result.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # 1. Regime-Adaptive Price-Volume Divergence
        # Calculate price and volume momentum
        price_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-6] - 1)
        volume_momentum = (current_data['volume'].iloc[-1] / current_data['volume'].iloc[-6] - 1)
        price_volume_divergence = price_momentum - volume_momentum
        
        # Assess volatility regime
        short_term_vol = current_data['returns'].iloc[-5:].std()
        long_term_vol = current_data['returns'].iloc[-20:].std()
        vol_ratio = short_term_vol / long_term_vol if long_term_vol > 0 else 1
        
        # Regime-specific scaling
        if vol_ratio > 1.2:
            regime_factor = 1.5  # High volatility regime
        elif vol_ratio < 0.8:
            regime_factor = 0.7  # Low volatility regime
        else:
            regime_factor = 1.0  # Normal regime
            
        regime_divergence = price_volume_divergence * regime_factor
        
        # 2. Momentum-Volume Acceleration
        # Calculate decaying momentum
        recent_returns = current_data['returns'].iloc[-5:]
        weights = np.exp(-np.arange(5) / 2.5)  # Exponential decay
        weighted_momentum = np.sum(recent_returns * weights) / np.sum(weights)
        
        # Volume acceleration (second derivative)
        if i >= 7:
            volume_changes = current_data['volume_change'].iloc[-4:]
            volume_acceleration = (volume_changes.iloc[-1] - volume_changes.iloc[-4]) / 3
        else:
            volume_acceleration = 0
            
        momentum_volume = weighted_momentum * volume_acceleration
        
        # 3. Range-Volume Confirmation
        # Range expansion
        current_range = current_data['true_range'].iloc[-1]
        avg_range = current_data['true_range'].iloc[-20:].mean()
        range_expansion = current_range / avg_range if avg_range > 0 else 1
        
        # Volume confirmation
        current_volume = current_data['volume'].iloc[-1]
        avg_volume = current_data['volume'].iloc[-20:].mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        range_confirmation = range_expansion * volume_ratio
        
        # 4. Volatility-Clustered Reversal
        # Return reversal
        recent_return = current_data['returns'].iloc[-1]
        prev_return = current_data['returns'].iloc[-2]
        reversal_signal = -recent_return if abs(recent_return) > 0.02 else 0
        
        # Volatility clustering adjustment
        vol_cluster_factor = vol_ratio if vol_ratio > 1 else 1/vol_ratio
        clustered_reversal = reversal_signal * vol_cluster_factor
        
        # Combine all components
        final_signal = (
            regime_divergence * 0.3 +
            momentum_volume * 0.25 +
            range_confirmation * 0.25 +
            clustered_reversal * 0.2
        )
        
        result.iloc[i] = final_signal
    
    # Fill any remaining NaN values
    result = result.fillna(0)
    
    return result
