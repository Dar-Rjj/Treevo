import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Amplitude Pressure Components
    df['amplitude_buy_pressure'] = (df['high'] - df['close']) * df['volume']
    df['amplitude_sell_pressure'] = (df['close'] - df['low']) * df['volume']
    
    # Volume-Price Divergence Analysis
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=5).mean()
    df['close_ma_5'] = df['close'].rolling(window=5, min_periods=5).mean()
    
    # Calculate volume-price correlation over 10-day window
    volume_price_corr = []
    for i in range(len(df)):
        if i >= 10:
            volume_window = df['volume'].iloc[i-9:i+1].values
            price_changes = (df['close'].iloc[i-9:i+1] - df['close'].iloc[i-10:i]).values
            if len(price_changes) == len(volume_window) and len(price_changes) >= 2:
                corr = np.corrcoef(volume_window, price_changes)[0, 1]
                if np.isnan(corr):
                    corr = 0
            else:
                corr = 0
        else:
            corr = 0
        volume_price_corr.append(corr)
    
    df['volume_price_correlation'] = volume_price_corr
    df['divergence_score'] = (1 - df['volume_price_correlation']) * (df['volume'] / df['volume_ma_5'])
    
    # Momentum Acceleration Dynamics
    df['momentum_3d'] = df['close'] - df['close'].shift(3)
    df['momentum_6d'] = df['close'] - df['close'].shift(6)
    df['momentum_acceleration'] = df['momentum_3d'] - df['momentum_6d']
    
    # Volatility Context Integration
    df['amplitude'] = df['high'] - df['low']
    df['amplitude_ma_5'] = df['amplitude'].rolling(window=5, min_periods=5).mean()
    
    # Calculate 10-day price volatility (std of Close returns)
    df['close_returns'] = df['close'].pct_change()
    df['price_volatility_10d'] = df['close_returns'].rolling(window=10, min_periods=10).std()
    
    # Calculate 10-day volume volatility (std of Volume)
    df['volume_volatility_10d'] = df['volume'].rolling(window=10, min_periods=10).std()
    df['regime_intensity'] = df['price_volatility_10d'] * df['volume_volatility_10d']
    
    # Component Synthesis
    df['net_amplitude_pressure'] = df['amplitude_buy_pressure'] - df['amplitude_sell_pressure']
    df['pressure_divergence_interaction'] = df['net_amplitude_pressure'] * df['divergence_score']
    df['momentum_enhanced_interaction'] = df['pressure_divergence_interaction'] * df['momentum_acceleration']
    
    # Volatility Scaling and Filtering
    df['amplitude_scaled_signal'] = df['momentum_enhanced_interaction'] * df['amplitude_ma_5']
    df['volatility_weighted_signal'] = df['amplitude_scaled_signal'] * df['regime_intensity']
    
    # Final Alpha Generation with Cubic Root Transformation
    alpha_factor = np.sign(df['volatility_weighted_signal']) * np.abs(df['volatility_weighted_signal']) ** (1/3)
    
    return alpha_factor
