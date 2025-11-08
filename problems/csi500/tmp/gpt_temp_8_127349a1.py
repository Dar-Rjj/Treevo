import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Regime-Adaptive Momentum with Volume-Price Dynamics
    """
    df = data.copy()
    
    # Calculate daily returns
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    
    # Multi-Timeframe Price Momentum
    df['ultra_short_momentum'] = df['close'] / df['close'].shift(1) - 1
    df['short_term_momentum'] = df['close'] / df['close'].shift(3) - 1
    df['medium_term_momentum'] = df['close'] / df['close'].shift(8) - 1
    df['momentum_acceleration'] = (df['short_term_momentum'] / df['medium_term_momentum']) - 1
    
    # Volatility Regime Classification
    df['ultra_short_vol'] = df['returns'].rolling(window=2).std()
    df['short_term_vol'] = df['returns'].rolling(window=5).std()
    df['volatility_ratio'] = df['ultra_short_vol'] / df['short_term_vol']
    
    # Volume-Price Dynamics Analysis
    df['volume_surge'] = df['volume'] / df['volume'].shift(2) - 1
    df['volume_trend'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_acceleration'] = df['volume_surge'] / df['volume_trend']
    
    # Calculate rolling correlation between price returns and volume
    rolling_corr = []
    for i in range(len(df)):
        if i >= 4:
            window_returns = df['returns'].iloc[i-3:i+1]  # t-3 to t
            window_volume = df['volume'].iloc[i-3:i+1]    # t-3 to t
            corr = np.corrcoef(window_returns, window_volume)[0, 1]
            rolling_corr.append(corr if not np.isnan(corr) else 0)
        else:
            rolling_corr.append(0)
    df['price_volume_corr'] = rolling_corr
    
    df['price_volume_divergence'] = np.abs(df['volume_acceleration']) * np.abs(df['price_volume_corr'])
    df['volume_confirmation_strength'] = np.abs(df['volume_acceleration'] * df['price_volume_corr'])
    
    # Initialize factor column
    df['factor'] = 0.0
    
    # Apply regime-based momentum selection
    for i in range(len(df)):
        if i < 8:  # Ensure we have enough data
            continue
            
        vol_ratio = df['volatility_ratio'].iloc[i]
        vol_conf_strength = df['volume_confirmation_strength'].iloc[i]
        
        # Regime-Based Momentum Selection
        if vol_ratio > 1.2:  # High Volatility
            momentum_factor = (0.6 * df['ultra_short_momentum'].iloc[i] + 
                             0.4 * df['short_term_momentum'].iloc[i])
        elif vol_ratio >= 0.8:  # Medium Volatility
            momentum_factor = (0.4 * df['short_term_momentum'].iloc[i] + 
                             0.6 * df['medium_term_momentum'].iloc[i])
        else:  # Low Volatility
            momentum_factor = (0.7 * df['medium_term_momentum'].iloc[i] + 
                             0.3 * df['momentum_acceleration'].iloc[i])
        
        # Volume Confirmation Multiplier
        vol_acc_corr = df['volume_acceleration'].iloc[i] * df['price_volume_corr'].iloc[i]
        
        if vol_conf_strength > 0.3:  # Strong Confirmation
            multiplier = 1 + vol_acc_corr
        elif vol_conf_strength > 0.1:  # Moderate Confirmation
            multiplier = 1 + 0.5 * vol_acc_corr
        else:  # Weak Confirmation
            multiplier = 1 + 0.2 * vol_acc_corr
        
        factor_value = momentum_factor * multiplier
        
        # Volatility-Adjusted Final Alpha
        if vol_ratio > 1.2:  # High Volatility
            final_factor = factor_value * (1 - vol_ratio)
        elif vol_ratio >= 0.8:  # Medium Volatility
            final_factor = factor_value
        else:  # Low Volatility
            final_factor = factor_value * (1 + vol_ratio)
        
        df.loc[df.index[i], 'factor'] = final_factor
    
    # Clean infinite values and replace with 0
    df['factor'] = df['factor'].replace([np.inf, -np.inf], 0)
    
    return df['factor']
