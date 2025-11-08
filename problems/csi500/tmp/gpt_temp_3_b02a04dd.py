import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Acceleration factor
    Combines price and volume momentum with acceleration signals and regime detection
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Core Momentum Components
    # Price Momentum
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Acceleration Signals
    # Price Acceleration
    data['price_accel_5d'] = data['price_momentum_5d'] - data['price_momentum_5d'].shift(1)
    data['price_accel_10d'] = data['price_momentum_10d'] - data['price_momentum_10d'].shift(1)
    data['price_accel_20d'] = data['price_momentum_20d'] - data['price_momentum_20d'].shift(1)
    
    # Volume Acceleration
    data['volume_accel_5d'] = data['volume_momentum_5d'] - data['volume_momentum_5d'].shift(1)
    data['volume_accel_10d'] = data['volume_momentum_10d'] - data['volume_momentum_10d'].shift(1)
    data['volume_accel_20d'] = data['volume_momentum_20d'] - data['volume_momentum_20d'].shift(1)
    
    # Simple Regime Detection
    # Volatility Regime
    data['current_range'] = (data['high'] - data['low']) / data['close']
    data['avg_range_5d'] = data['current_range'].rolling(window=5).mean()
    data['high_volatility'] = data['current_range'] > data['avg_range_5d']
    
    # Volume Regime
    data['volume_ratio'] = data['volume'] / data['volume'].shift(1)
    data['high_volume'] = data['volume_ratio'] > 1.5
    
    # Clear Divergence Patterns
    # Price-Volume Divergence for 5-day momentum
    data['sign_match_5d'] = np.sign(data['price_momentum_5d']) == np.sign(data['volume_momentum_5d'])
    data['strength_ratio_5d'] = np.abs(data['price_momentum_5d']) / (np.abs(data['volume_momentum_5d']) + 1e-8)
    
    # Timeframe Consistency
    accel_signs = pd.DataFrame({
        'accel_5d': np.sign(data['price_accel_5d']),
        'accel_10d': np.sign(data['price_accel_10d']),
        'accel_20d': np.sign(data['price_accel_20d'])
    })
    data['direction_agreement'] = accel_signs.apply(lambda x: (x == x.mode()[0]).sum() if len(x.mode()) > 0 else 1, axis=1)
    data['strength_agreement'] = data[['price_accel_5d', 'price_accel_10d', 'price_accel_20d']].abs().mean(axis=1)
    
    # Regime-Adaptive Weighting
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if pd.isna(data.iloc[i]['price_momentum_5d']):
            continue
            
        # Determine regime
        high_vol = data.iloc[i]['high_volatility']
        high_vol_vol = data.iloc[i]['high_volume']
        
        # Calculate base components
        price_momentum_avg = np.nanmean([data.iloc[i]['price_momentum_5d'], 
                                       data.iloc[i]['price_momentum_10d'], 
                                       data.iloc[i]['price_momentum_20d']])
        
        volume_momentum_avg = np.nanmean([data.iloc[i]['volume_momentum_5d'], 
                                        data.iloc[i]['volume_momentum_10d'], 
                                        data.iloc[i]['volume_momentum_20d']])
        
        price_accel_avg = np.nanmean([data.iloc[i]['price_accel_5d'], 
                                    data.iloc[i]['price_accel_10d'], 
                                    data.iloc[i]['price_accel_20d']])
        
        volume_accel_avg = np.nanmean([data.iloc[i]['volume_accel_5d'], 
                                     data.iloc[i]['volume_accel_10d'], 
                                     data.iloc[i]['volume_accel_20d']])
        
        # Apply regime-specific weighting
        if high_vol and high_vol_vol:
            # High Volatility + High Volume: Focus on volume acceleration
            base_signal = 0.7 * volume_accel_avg + 0.3 * price_accel_avg
        elif high_vol and not high_vol_vol:
            # High Volatility + Normal Volume: Balanced acceleration
            base_signal = 0.6 * price_accel_avg + 0.4 * volume_accel_avg
        elif not high_vol and high_vol_vol:
            # Normal Volatility + High Volume: Focus on volume momentum
            base_signal = 0.6 * volume_momentum_avg + 0.4 * price_momentum_avg
        else:
            # Normal Volatility + Normal Volume: Focus on price momentum
            base_signal = 0.7 * price_momentum_avg + 0.3 * volume_momentum_avg
        
        # Incorporate divergence pattern strength
        divergence_multiplier = 1.0
        if not data.iloc[i]['sign_match_5d']:
            # Penalize divergence
            divergence_multiplier = 0.7
        
        # Apply timeframe consistency multiplier
        consistency_multiplier = data.iloc[i]['direction_agreement'] / 3.0
        
        # Calculate final signal
        final_signal = base_signal * divergence_multiplier * consistency_multiplier
        
        # Volatility scaling
        volatility_scale = data.iloc[i]['current_range'] + 1e-8
        scaled_signal = final_signal / volatility_scale
        
        alpha_signal.iloc[i] = scaled_signal
    
    # Clean up and return
    alpha_signal = alpha_signal.replace([np.inf, -np.inf], np.nan)
    return alpha_signal
